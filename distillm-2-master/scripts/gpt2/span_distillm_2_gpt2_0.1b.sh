#!/bin/bash
# ==========================================
# Run DistillM training on a specific GPU
# Model: GPT2-1.5B → GPT2-0.1B
# ==========================================

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# --- Accelerate launch ---
accelerate launch \
  --config_file distillm-2-master/accelerate_configs/deepspeed_zero0.yaml \
  --num_processes=1 \
  distillm-2-master/src/run_span_distillm.py \
  distillm-2-master/training_configs/gpt2-1.5b-0.1b-span-distillm2.yaml