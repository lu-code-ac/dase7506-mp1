"""Optimized training script with Cosine Warmup LR Schedule.

Adds label smoothing and EMA weight averaging on top of the baseline recipe.
- checkpoint.pt      : EMA weights  (primary submission / evaluation)
- checkpoint_raw.pt  : raw weights  (for ablation / debugging)
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def configure_optimizers(model, weight_decay=0.1, learning_rate=0.003, betas=(0.9, 0.95)):
    decay_params = []
    nodecay_params = []
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_params:
            continue
        seen_params.add(id(param))

        if param.dim() < 2 or 'bias' in name or 'ln' in name or 'norm' in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


@torch.no_grad()
def ema_update(ema_state, model_state, decay):
    """In-place EMA: ema = decay * ema + (1 - decay) * model."""
    for k, v in model_state.items():
        if v.dtype.is_floating_point:
            ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema_state[k].copy_(v)


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--peak-lr', type=float, default=0.003)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--label-smoothing', type=float, default=0.1,
                   help='Label smoothing for cross entropy; set 0 to disable.')
    p.add_argument('--ema-decay', type=float, default=0.999,
                   help='EMA decay; set 0 to disable EMA.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    peak_lr = args.peak_lr
    optimizer = configure_optimizers(model, weight_decay=args.weight_decay, learning_rate=peak_lr)

    use_ema = args.ema_decay > 0
    if use_ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.

    # 动态适配总步数的 5% Warmup，最少 100 步
    warmup_steps = max(100, int(args.steps * 0.05))

    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]

        # Cosine warmup + decay
        if step < warmup_steps:
            learning_rate = peak_lr * (step + 1) / warmup_steps
        else:
            decay_ratio = (step - warmup_steps) / max(1, args.steps - warmup_steps)
            learning_rate = peak_lr * 0.05 + 0.5 * (peak_lr * 0.95) * (1.0 + math.cos(math.pi * decay_ratio))

        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:, :-1]).flatten(0, 1).float()
            loss = F.cross_entropy(
                logits, batch[:, 1:].flatten(),
                label_smoothing=args.label_smoothing,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if use_ema:
            # 前 10 步让 decay 从较小值平滑爬升，避免初期抖动
            decay = min(args.ema_decay, (step + 1) / (step + 10))
            ema_update(ema_state, model.state_dict(), decay)

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(), 'lr': learning_rate,
                   'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            # 同时记录 raw 和 EMA 的验证 BPB，便于选 checkpoint
            raw_metrics = score(model, *data['validation'], device, 'fp32')
            raw_metrics.pop('window_nll_nats')
            validation_history.append({'step': step + 1, 'weights': 'raw', **raw_metrics})
            print(json.dumps({'validation_raw': validation_history[-1]}), flush=True)
            intermediate_validation_seconds += raw_metrics['seconds']

            if use_ema:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
                ema_metrics = score(model, *data['validation'], device, 'fp32')
                ema_metrics.pop('window_nll_nats')
                model.load_state_dict(backup)
                validation_history.append({'step': step + 1, 'weights': 'ema', **ema_metrics})
                print(json.dumps({'validation_ema': validation_history[-1]}), flush=True)
                intermediate_validation_seconds += ema_metrics['seconds']

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds

    # 用 raw 权重做一次验证
    validation_raw = score(model, *data['validation'], device, 'fp32')
    validation_raw.pop('window_nll_nats')

    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 用 EMA 权重做一次验证
    if use_ema:
        model.load_state_dict(ema_state)
        validation_ema = score(model, *data['validation'], device, 'fp32')
        validation_ema.pop('window_nll_nats')
    else:
        validation_ema = None

    # 提交用 checkpoint：优先 EMA
    save_state = ema_state if use_ema else raw_state
    checkpoint = args.run_dir / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'model': {k: v.cpu() for k, v in save_state.items()},
        'seed': args.seed,
        'train_tokens': args.steps * args.batch_size * 256,
        'weights': 'ema' if use_ema else 'raw',
    }, checkpoint)

    # 消融用 checkpoint：raw
    checkpoint_raw = args.run_dir / 'checkpoint_raw.pt'
    torch.save({
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'model': {k: v.cpu() for k, v in raw_state.items()},
        'seed': args.seed,
        'train_tokens': args.steps * args.batch_size * 256,
        'weights': 'raw',
    }, checkpoint_raw)

    result = {
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'seed': args.seed,
        'parameters': sum(p.numel() for p in model.parameters()),
        'precision': precision,
        'train_tokens': args.steps * args.batch_size * 256,
        'batch_size': args.batch_size,
        'peak_lr': peak_lr,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'ema_decay': args.ema_decay if use_ema else None,
        'warmup_steps': warmup_steps,
        'preparation_seconds': preparation_seconds,
        'train_seconds': train_seconds,
        'validation_raw': validation_raw,
        'validation_ema': validation_ema,
        'history': history,
        'validation_history': validation_history,
        'intermediate_validation_seconds': intermediate_validation_seconds,
        'process_seconds': time.perf_counter() - total_started,
        'torch_version': str(torch.__version__),
        'threads': args.threads,
        'checkpoint_sha256': sha(checkpoint),
        'checkpoint_raw_sha256': sha(checkpoint_raw),
        'implementation_sha256': implementation_sha,
        **device_metrics(device),
    }
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()