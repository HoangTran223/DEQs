#! /bin/bash
# ========================================================
# DEQ + DistiLLM: Qwen1.5-1.8B (teacher) -> DEQ-Qwen (student)
# ========================================================

GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export CUDA_HOME=$CONDA_PREFIX

MASTER_ADDR=localhost
MASTER_PORT=50$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=./distillm-fdd
CKPT="./models/qwen_0.5b"
TEACHER_CKPT="./models/qwen_1.5b_sft"
CKPT_NAME="qwen-0.5b"
TEACHER_CKPT_NAME="qwen-1.5b-sft"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen/"
# hp
BATCH_SIZE=2
LR=0.00005
GRAD_ACC=4
EVAL_BATCH_SIZE=8
EPOCHS=10
# length
MAX_LENGTH=256
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen/train/deq_0.5B_1.5B_norm"
SEED=42

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 200"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 0.5"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 0.8"
OPTS+=" --warmup-ratio 0.1"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 128"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 1000"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# distillation type
OPTS+=" --type srkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# DEQ
OPTS+=" --n-deq-blocks 2"
OPTS+=" --deq-f-max-iter 15"
OPTS+=" --deq-b-max-iter 15"
OPTS+=" --deq-f-tol 1e-3"
OPTS+=" --deq-b-tol 1e-6"
OPTS+=" --deq-n-states 1"
OPTS+=" --deq-gamma 0.9"
OPTS+=" --deq-solver fixed_point_iter"
OPTS+=" --deq-norm-type spectral_norm"
OPTS+=" --deq-init-pretrained"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/deq_finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
