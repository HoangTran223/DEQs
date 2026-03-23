# DEQ Student for LLM Knowledge Distillation

DEQ (Deep Equilibrium Model) student replaces the stack of N decoder layers with 1-2 weight-tied layers solved via fixed-point iteration — drastically fewer parameters while retaining expressive power.

**Server**: 2x NVIDIA A40 46GB, CUDA 13.0, Ubuntu

---

## 1. Install

```bash
conda activate deq

# Base deps (if not already installed via distillm-fdd/distillm-2-master)
pip install deepspeed accelerate datasets peft nltk numerize rouge-score sentencepiece protobuf rich spacy
python -m spacy download en_core_web_sm

# torchdeq (from local repo)
pip install /mnt/hoangtv/DEQs/torchdeq
```

## 2. Directory layout

```
KD/
├── distillm-fdd/                 # DistiLLM + DEQ framework
│   ├── deq_model.py              # DEQ student model (GPT-2 / Qwen / OPT)
│   ├── deq_finetune.py           # Training script (DistiLLM + DEQ)
│   ├── processed_data/dolly/     # Preprocessed training data
│   └── scripts/qwen/deq/         # Shell scripts
├── distillm-2-master/            # DistiLLM-2 + DEQ framework
│   ├── src/run_deq_distillm.py   # Training script (DistiLLM-2 + DEQ)
│   ├── training_configs/         # YAML configs
│   └── data/reformatted/gpt2/    # DPO-format data
├── models/                       # Local model checkpoints
│   ├── qwen_0.5b/                # Student init
│   └── qwen_1.5b_sft/           # Teacher (SFT)
├── torchdeq/                     # torchdeq library (pip install from here)
├── data/                         # Eval benchmarks (dolly, vicuna, self-inst, sinst)
└── run_deq_example.sh            # Run all DEQ experiments
```

## 3. Run

### DistiLLM + DEQ (Qwen 0.5B → DEQ)

```bash
cd /mnt/hoangtv/DEQs/KD
bash distillm-fdd/scripts/qwen/deq/train_0.1B_1.5B.sh
```

### DistiLLM-2 + DEQ

```bash
bash distillm-2-master/scripts/gpt2/deq_distillm_2_gpt2_0.1b.sh
```

## 4. Key hyperparameters

| Param | Default | Description |
|-------|---------|-------------|
| `--n-deq-blocks` | 2 | Blocks inside f_theta (1 or 2) |
| `--deq-f-max-iter` | 10 | Fixed-point solver iterations |
| `--deq-f-tol` | 1e-3 | Convergence tolerance |
| `--deq-norm-type` | weight_norm | Normalization for stability |
| `--deq-init-pretrained` | flag | Init blocks from pretrained weights |
| `--model-type` | qwen | Model family: `gpt2`, `qwen`, `opt` |
| `--kd-ratio` | 0.5 | Weight of KD loss vs CE loss |

---

## 5. Experiment observations (Qwen 0.5B student, Qwen 1.5B teacher)

### Baseline DistiLLM (full Qwen 0.5B, 463M params)
- rougeL: 15.5 → **21.5** (10 epochs)
- Loss: 2.19 → 0.99

### DEQ experiments

| Run | LR | Norm | kd_ratio | Epochs trained | Best rougeL | Notes |
|-----|-----|------|----------|----------------|-------------|-------|
| `deq_0.5B_1.5B_norm` | 5e-5 | spectral | 0.5 | 5 | **9.80** | Stable, but rougeL peaks epoch 2 then drops |
| `deq_0.5B_1.5B_norm_0.005` | 5e-3 | spectral | 0.5 | 6 | 6.22 | LR too high, avg_loss spike to 179 at epoch 1 |
| Earlier (NaN crash) | 1e-4 | weight | 1.0 | 3 | 2.88 | NaN at epoch 3, kd_ratio=1.0 was the problem |

### Key findings

1. **fp16 causes NaN in DEQ**: The fixed-point solver iterates the same block 10-30 times. In fp16, numerical errors accumulate across iterations → NaN logits. **Fixed**: DEQ solver now runs blocks in fp32 during the iteration loop, converts back to fp16 for the final output.

2. **kd_ratio=1.0 is fatal for DEQ**: With 100% KD loss and 0% CE loss, the DEQ student (starting from weak representations) cannot match the teacher distribution. Always use `kd_ratio ≤ 0.5` to include ground-truth CE loss.

3. **spectral_norm destroys pretrained weights**: Spectral norm forces the spectral norm of every linear layer to exactly 1. If pretrained weights had spectral norm = 5-10, this divides all weights by 5-10x, destroying the pretrained representations. **Recommendation**: Use `weight_norm` which is gentler, or `spectral_norm` with `norm_clip_value > 1`.

4. **LR sensitivity**: DEQ is more sensitive to learning rate than standard models. Best range: **1e-5 to 5e-5**. Higher LR (5e-3) causes loss spikes and instability.

5. **Performance gap**: Best DEQ (9.80 rougeL) vs baseline (21.5 rougeL) — significant gap. The DEQ student has ~182M params (39% of original 463M). The 2 blocks (from layers 0,1 of pretrained) were trained for sequential single-pass processing, not iterative fixed-point convergence. This is the fundamental challenge.

6. **Overfitting pattern**: DEQ rougeL peaks at epoch 2-3 then drops while training loss continues decreasing. Early stopping or larger dataset may help.
