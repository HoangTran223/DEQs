#!/bin/bash
# ==========================================
# DistiLLM-2 + DEQ: Qwen 1.5B SFT -> DEQ-Qwen
# ==========================================

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$CONDA_PREFIX

accelerate launch \
  --config_file distillm-2-master/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  distillm-2-master/src/run_deq_distillm.py \
  distillm-2-master/training_configs/gpt2-1.5b-deq-distillm2.yaml \
  --n_deq_blocks 2 \
  --deq_f_max_iter 30 \
  --deq_b_max_iter 30 \
  --deq_f_tol 1e-3 \
  --deq_b_tol 1e-6 \
  --deq_n_states 1 \
  --deq_gamma 0.8 \
  --deq_solver fixed_point_iter \
  --deq_norm_type weight_norm \
  --deq_init_pretrained \
  --deq_model_type qwen
