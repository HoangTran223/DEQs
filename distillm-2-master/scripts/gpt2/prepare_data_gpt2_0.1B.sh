#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="openai-community/gpt2"
OUTPUT_DIR="${BASE_PATH}/data/dpo/${MODEL_PATH}"


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
python distillm-2-master/generate/generate.py ${OPTS} >> ${OUTPUT_DIR}/eval.log 2>&1
