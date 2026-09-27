import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads, dropout=0.0, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.c_attn = nn.Linear(width, 3 * width, bias=False)
        self.c_proj = nn.Linear(width, width, bias=False)
        self.c_proj.is_residual = True
        self.dropout = dropout
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        qkv = self.c_attn(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(width, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, width, bias=False)
        self.w3 = nn.Linear(width, hidden_dim, bias=False)
        self.w2.is_residual = True

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, width, heads, hidden_dim, dropout=0.0, qk_norm=True):
        super().__init__()
        self.ln_1 = RMSNorm(width)
        self.attn = CausalSelfAttention(width, heads, dropout, qk_norm)
        self.ln_2 = RMSNorm(width)
        self.mlp = SwiGLU(width, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        x = x + self.drop(self.attn(self.ln_1(x), freqs_cis))
        x = x + self.drop(self.mlp(self.ln_2(x)))
        return x


class StudentModel(nn.Module):
    def __init__(self, vocab=2048, width=256, heads=4, depth=4,
                 hidden_dim=None, context=256, dropout=0.1, qk_norm=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * width / 3)  # ≈ 680

        self.context = context
        self.tok_emb = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([
            Block(width, heads, hidden_dim, dropout, qk_norm) for _ in range(depth)
        ])
        self.ln_f = RMSNorm(width)

        freqs_cis = precompute_freqs_cis(width // heads, context, theta=10000.0)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model Check] params={n:,} | width={width} heads={heads} "
              f"head_dim={width//heads} hidden={hidden_dim} depth={depth} "
              f"dropout={dropout} qk_norm={qk_norm}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, 'is_residual', False):
                std *= (2 * len(self.layers)) ** -0.5
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        freqs_cis = self.freqs_cis[:T]
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.ln_f(x)
        return F.linear(x, self.tok_emb.weight)  # weight tying

    def predict_log_probs(self, idx):
        return F.log_softmax(self.forward(idx).float(), dim=-1)


def build_model(config_or_vocab=None, **kwargs):
    vocab = 2048
    if isinstance(config_or_vocab, dict):
        vocab = config_or_vocab.get("vocab", 2048)
    elif isinstance(config_or_vocab, int):
        vocab = config_or_vocab
    return StudentModel(vocab=vocab, width=256, heads=4, depth=4,
                        hidden_dim=680, context=256, dropout=0.1, qk_norm=True)