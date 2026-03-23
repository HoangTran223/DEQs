#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="distillm-2-master/outputs/gpt2-0.1b-span-distillm2/checkpoint-7145"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

OPTS+=" --val_batch_size 128"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path "
OPTS+=" --tokenizer openai-community/gpt2"

# ==== Gọi Python ====
python src/run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log 2>&1
