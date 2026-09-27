# DASE7506 MP1 — Small Language Model Challenge

**Author:** [Xu Lu / 3036804397]
**Final full-test BPB (FP32, CPU):** **1.5224891792321562**
**Commit:** `b07d1c712aafcaee4e8d13e6c0128e6e66f18cd8`

---

## 1. Introduction

The goal of this assignment is to train a small language model from scratch on the provided WikiText-2 training split, using the supplied BPE-2048 tokenizer, and to minimize full-test bits per byte (BPB) under three evaluation constraints: at most 5× the baseline CPU scoring time, at most 4 GiB peak evaluation RAM, and at most 64 MiB of uncompressed inference assets.  

The provided baseline GPT (four blocks, width 128, 1,088,256 parameters) achieves approximately 2.10 test BPB. Our submission reaches a final **full-test BPB of 1.5224891792321562** while using only 2.5× the baseline CPU scoring time and a 14 MB checkpoint, comfortably inside all limits. 

The improvement is achieved by (i) a wider and better-proportioned architecture, (ii) a stronger training recipe (label smoothing, cosine schedule with warmup, EMA), and (iii) training on roughly 17× more targets than the baseline recipe. We also report an ablation that isolates the contribution of each component.

---

## 2. Method

### 2.1 Model architecture

The submitted model is a decoder-only transformer with the following configuration:

| Component               | Baseline         | Submitted                    |
| ----------------------- | ---------------- | ---------------------------- |
| Blocks (depth)          | 4                | 4                            |
| Width                   | 128              | 256                          |
| Attention heads         | 4                | 4                            |
| Head dimension          | 32               | 64                           |
| FFN type                | GELU, 4×         | SwiGLU, 8/3×                 |
| FFN hidden dim          | 512              | 680                          |
| Normalization           | LayerNorm        | RMSNorm (pre-norm)           |
| Positional encoding     | Learned absolute | RoPE                         |
| Attention normalization | None             | QK-Norm (RMSNorm on Q and K) |
| Vocabulary projection   | Separate         | Weight-tied with embedding   |
| Dropout                 | 0                | 0.1                          |
| Parameters              | 1,088,256        | 3,664,640                    |

**Rationale for each change:**

- **Width 256, head_dim 64.** The baseline uses head_dim 32, which is unusually small; attention at this scale has limited capacity per head. Moving to head_dim 64 is standard in modern small transformers and gives the attention layers more expressive queries and keys.
- **SwiGLU with hidden ratio 8/3.** The baseline uses GELU with a 4× expansion. SwiGLU is consistently stronger in the literature at matched parameter count, and the 8/3 ratio keeps parameter count roughly equal to the GELU 4× variant while adding the multiplicative gating interaction.
- **RMSNorm pre-norm.** Faster and as effective as LayerNorm in small transformers, and cheaper to compute.
- **RoPE.** Removes learned positional embeddings and provides smoother relative-position behavior, which is especially helpful when training token budgets are modest.
- **QK-Norm.** Normalizing queries and keys before RoPE stabilizes attention logits, which allows a larger effective learning rate and faster convergence early in training.
- **Weight tying.** Ties the output projection to the token embedding, which removes 524K parameters and improves generalization on small-data regimes.
- **Dropout 0.1.** With 164M training tokens on a 3.7M-parameter model, the effective number of epochs on WikiText-2 is large, and dropout measurably reduces overfitting.

### 2.2 Training recipe

| Hyperparameter         | Value                                             |
| ---------------------- | ------------------------------------------------- |
| Optimizer              | AdamW (β₁=0.9, β₂=0.95)                           |
| Weight decay           | 0.1 (applied to weight matrices only)             |
| Peak learning rate     | 3e-3                                              |
| LR schedule            | 5% linear warmup, then cosine decay to 5% of peak |
| Warmup steps           | 2000                                              |
| Batch size             | 16                                                |
| Steps                  | 40,000                                            |
| Sequence length        | 256                                               |
| Total training targets | 163,840,000                                       |
| Mixed precision        | BF16                                              |
| Gradient clipping      | 1.0 (global L2 norm)                              |
| Label smoothing        | 0.1                                               |
| EMA decay              | 0.999                                             |
| Seed                   | 17                                                |

The learning rate schedule uses a 5% warmup (2000 steps) followed by a cosine decay from peak down to 5% of peak. The final learning rate is 1.5e-4. This is a standard formulation for small transformers and we did not tune it.

**Label smoothing** at ε=0.1 replaces the one-hot target with a mixture of 0.9 on the true token and 0.1 spread uniformly. This improves calibration and slightly reduces overfitting.

**EMA weight averaging** maintains a shadow copy of the model parameters that is updated as `θ_ema ← decay · θ_ema + (1 − decay) · θ_model` at every step. At the end of training, the EMA weights are used for evaluation. This effectively ensembles the last ~1000 steps of training and consistently reduces validation BPB by 0.01 in our setup.

---

## 3. Experimental Setup

All experiments used the same WikiText-2 training split, the same BPE-2048 tokenizer, and the same evaluation protocol (`7506-mp1-wt2-v2`). Model selection used only the validation split. The full test split was evaluated exactly once, after the method was frozen.

### 3.1 Baseline

The provided baseline model (`model.py`, width 128, depth 4, GELU MLP, 1,088,256 parameters) was trained with the baseline recipe (1,200 steps, batch 32) and evaluated under the same scorer. On this machine it reaches:

| Metric                        | Value  |
| ----------------------------- | ------ |
| Baseline validation BPB (raw) | 1.9547 |
| Baseline validation BPB (EMA) | 1.9597 |
| Baseline CPU evaluation time  | 4.99 s |

The baseline CPU time of 4.99 s defines the 5× budget of 24.94 s for this machine.

### 3.2 Ablation

Ablation runs B, C, and D used the same budget (40,000 steps × 16 batch × 256 targets = 163.84M tokens) and were evaluated on the validation split only. Row A is the provided baseline model trained with its original recipe (1,200 steps, batch 32 = 9.83M tokens).

| Variant              | Architecture | Label smoothing | EMA   | Validation BPB |
| -------------------- | ------------ | --------------- | ----- | -------------- |
| A. Baseline          | `model.py`   | 0.1             | 0.999 | 1.9597         |
| B. New architecture  | `student.py` | 0               | —     | 1.5165         |
| C. + label smoothing | `student.py` | 0.1             | —     | 1.5149         |
| D. + EMA (submitted) | `student.py` | 0.1             | 0.999 | **1.5048**     |

Interpretation:

- **A → B: −0.443 BPB.** The combined effect of widening the model, changing FFN type, adding QK-Norm, using RoPE, adding dropout, and training 17× longer. This is by far the dominant contribution.
- **B → C: −0.0016 BPB.** Label smoothing alone barely moves the metric at this training budget; its regularization value is small once dropout is already in place.
- **C → D: −0.0101 BPB.** EMA provides a small but reliable gain at essentially zero evaluation cost, since the EMA weights are simply a different point in the same parameter space.

### 3.3 Final test evaluation

The final submitted checkpoint (EMA weights from run D) was evaluated on the full test split with FP32 on CPU:

| Metric                | Value               |
| --------------------- | ------------------- |
| Test BPB              | 1.5224891792321562  |
| Test token perplexity | 24.11111799402677   |
| Test targets          | 428,405             |
| Test UTF-8 bytes      | 1,292,013           |
| CPU evaluation time   | 12.307940699975006s |

Validation BPB (≈1.5048) and test BPB (≈1.5225) differ by 0.018, which is the typical generalization gap for this benchmark.

---

## 4. Results and Resource Compliance

| Constraint           | Limit                | Submitted                  | Status   |
| -------------------- | -------------------- | -------------------------- | -------- |
| CPU scoring time     | ≤ 5 × 4.99 = 24.94 s | 12.307940699975006s        | ✓ (2.5×) |
| Peak evaluation RAM  | ≤ 4 GiB              | well below                 | ✓        |
| Inference assets     | ≤ 64 MiB             | 13.99 MB (FP32 checkpoint) | ✓        |
| Evaluation precision | FP32                 | FP32                       | ✓        |
| Evaluation device    | CPU reproducible     | CPU                        | ✓        |

The checkpoint is 13.99 MB on disk, well under the 64 MiB asset cap. No retrieval database, cached validation/test answers, or cross-window state is used; the model is a pure feed-forward predictor over the supplied tokenizer.

---

## 5. Discussion

### 5.1 What worked and why

The dominant factor in our improvement is the combination of a wider architecture with a longer training budget. The baseline is trained for only 1,200 steps, which underfits WikiText-2 by a significant margin. Training the same recipe for 40,000 steps alone would improve the baseline substantially; the architectural changes amplify this.  

Among architectural changes, the most consequential are likely:

- **head_dim 32 → 64** and **width 128 → 256**, which directly expand the per-head attention capacity.
- **SwiGLU** instead of GELU, which gives the FFN a multiplicative interaction at matched parameter count.
- **QK-Norm**, which stabilizes attention and enables stable training at the larger effective learning rate that a wider model can sustain.

### 5.2 What barely mattered

Label smoothing contributed only 0.0016 BPB at this scale. Once dropout is already in place and the model is not heavily overfitting, label smoothing has little left to regularize. We kept it because it does not hurt and because it slightly improves calibration.

### 5.3 Trade-offs

The submitted model uses about 3.5× the per-token FLOPs of the baseline, which is the direct cause of the 2.5× CPU scoring time. This is well within the 5× budget, but leaves room for further scaling: we could increase width to 320 or depth to 5 and still fit within the limit.

The training cost is also modest: about 25 minutes on an RTX 3050 Ti Laptop GPU for 163.84M training tokens. This is the entire training budget of the submission; no pretrained weights or external data are used.

### 5.4 Limitations

- We did not explore larger widths (320, 384) or depth 5, which could plausibly improve BPB further within the CPU budget.
- We trained with a single seed (17). Multi-seed selection using validation would likely provide another 0.005–0.01 BPB at the cost of additional training time.
- The training-token budget was chosen to fit a 25-minute GPU window; Longer training would likely still be on the improving part of the curve, though the last 4,000 steps changed validation BPB by less than 0.001, suggesting convergence.

---

## 6. Reproduction

### 6.1 Environment

- Python 3.12
- PyTorch 2.7.1+cu126 (CUDA 12.6 build; tested on a CUDA 12.3 driver)
- numpy 2.5.3, tokenizers 0.21.4
- Training device: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Evaluation device: CPU, FP32

### 6.2 Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install numpy==2.5.3 tokenizers==0.21.4
```

### 6.3 Evaluate the submitted checkpoint (no retraining)

```
.venv/Scripts/python.exe evaluate.py \
  --checkpoint runs/student_v1_main/checkpoint.pt \
  --split test \
  --device cpu \
  --precision fp32
```

Expected result: `bpb = 1.5224891792321562`, `seconds = 12.307940699975006`.

### 6.4 Retrain from scratch

```
.venv/Scripts/python.exe train.py \
  --implementation student \
  --device cuda \
  --seed 17 \
  --steps 40000 \
  --batch-size 16 \
  --eval-every 4000 \
  --peak-lr 0.003 \
  --label-smoothing 0.1 \
  --ema-decay 0.999 \
  --run-dir runs/student_v1_main
```

On CPU, replace `--device cuda` with `--device cpu --threads 4`.

### 6.5 Frozen artifacts

- `student.py`: SHA256 `3a74346a2192f92f769de4260aa9e7105da6aafdd5f04872b98511908e766f18`
- `train.py`: SHA256 `b4cf41296f3b0f06d10cdaf637e78b9a400d39a9551d4266eaeca4ecaa7a83da`
- Submitted checkpoint SHA256 (EMA): `c36d7f24211dce2e2b8deb60189c706cc4b055ce279d013c9275a4cd22c7a56f`
- Commit: `b07d1c712aafcaee4e8d13e6c0128e6e66f18cd8`

## 7. AI Assistance Disclosure

Parts of `student.py` and `train.py` were developed with the help of  AI assistants (ds and gemini). Specifically: EMA weight averaging, QK-Norm,label smoothing, the cosine LR schedule parameters, the SwiGLU hidden-dimension ratio, and the overall training recipe. All code was reviewed, understood, tested, and verified by the author. All experiments, seeds, and reported scores were produced by the author.