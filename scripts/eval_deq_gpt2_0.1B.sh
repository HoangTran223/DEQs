#! /bin/bash
# ========================================================
# Evaluate DEQ Student: load trained checkpoint and run
# generation + ROUGE-L on the dev set
# ========================================================

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=./distillm-fdd
# Point model-path to the trained DEQ checkpoint
CKPT_STEP=3570
MODEL_PATH="${BASE_PATH}/results/gpt2/train/deq_0.1B_1.5B/${CKPT_STEP}"
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
SAVE_PATH="${BASE_PATH}/results/gpt2/eval/deq_0.1B_1.5B/${CKPT_STEP}"

OPTS=""
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --model-type gpt2"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
OPTS+=" --batch-size 1"
OPTS+=" --eval-batch-size 64"
OPTS+=" --max-length 256"
OPTS+=" --max-prompt-length 128"
OPTS+=" --do-eval"
OPTS+=" --eval-gen"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 42"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type eval"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

mkdir -p ${SAVE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/deq_finetune.py ${OPTS} $@"
echo ${CMD}
${CMD}
