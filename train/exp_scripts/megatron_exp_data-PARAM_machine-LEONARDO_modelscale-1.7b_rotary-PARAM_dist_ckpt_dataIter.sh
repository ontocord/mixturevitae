nodes=$1
tokens=$2
micro_bs=$3
gas=$4
seq_length=$5
rotary_base=$6
lr_schedule=$7
lr=$8
lr_min=$9
lr_warmup_iters=${10}
save_interval_iter=${11}
data_name=${12}
data_switch=${13}
data_chunks=${14}

machine_name="MACHINE"
model="1.7b"
data="${data_name}" # FineWeb-Edu-1.4T, SlimPajama
tokenizer="${data_switch}"

tp=1

billion_num=1000000000 # 1B
tokens_billion=$(( tokens / billion_num ))
tokens_num_string="${tokens_billion}B"

num_gpus_per_node=4
total_num_gpus=$((nodes*num_gpus_per_node))

global_bs=$(((total_num_gpus*micro_bs*gas)/tp))

name_template="open-sci-ref_model-${model}_data-${data}_tokenizer-${tokenizer}_samples-${tokens_num_string}_global_bs-${global_bs}_context-${seq_length}_rotary-${rotary_base}_schedule-${lr_schedule}_lr-${lr}_warmup-${lr_warmup_iters}_machine-${machine_name}"

template_script_path="${core_path}/template_scripts"

mkdir -p ${template_script_path}

echo "#!/bin/bash

#SBATCH --nodes=${nodes}
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --threads-per-core=1
#SBATCH --output=<slurm_out_path>/${name_template}_%j.out" > ${template_script_path}/exp_${name_template}.sbatch

echo '# For large node numbers (> 64 nodes): 

MICRO_BATCH_SIZE=$1
GAS=$2
LR_SCHEDULE=$3
LR=$4
LR_MIN=$5
LR_WARMUP_ITERS=$6
TOKENS_TOTAL=$7
SEQ_LENGTH=$8
ROTARY_BASE=$9
SAVE_INTERVAL_ITERS=${10}
DATA_NAME=${11}
DATA_SWITCH=${12}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MEGATRON_CACHE_FOLDER="${MEGATRON_CACHE_BASE}/${USER}"
mkdir -p ${MEGATRON_CACHE_FOLDER}

export MEGATRON_CACHE="${MEGATRON_CACHE_FOLDER}/MEGATRON_CACHEDIR"
mkdir -p $MEGATRON_CACHE
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p $TENSORBOARD_DIR

export APPTAINER_CACHEDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_TMPDIR"

mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

IMAGE=${SHARED_CONTAINERS}/pytorch_24.09-py3_leonardo.sif

export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=120

GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR="${MASTER_ADDR}"' >> ${template_script_path}/exp_${name_template}.sbatch

GREP_STRING="grep -oP '(?<=Address: ).*')" # grep -oP '(?<=Address: ).*')
echo 'MASTER_IP="$(nslookup "$MASTER_ADDR" |'${GREP_STRING}\" >> ${template_script_path}/exp_${name_template}.sbatch

echo 'echo $MASTER_IP
export MASTER_ADDR=$MASTER_IP
MASTER_PORT=12345
NNODES=$SLURM_NNODES

export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "CHUNKS=${data_chunks} # number of chunks to go through for DCLM, 0..6" >> ${template_script_path}/exp_${name_template}.sbatch

echo 'DATA_PATH=""

for ((i = 0; i <= CHUNKS ; i++  ))
do 
    PART_PATH="$DATA/tokenized/${DATA_NAME}/${DATA_SWITCH}/merged_${i}"
    DATA_PATH="${DATA_PATH} ${PART_PATH}"
done

TOKENIZER_TYPE="HuggingFaceTokenizer"

DATA_NUM_WORKERS=4
DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type $TOKENIZER_TYPE
    --split 989,10,1
    --num-workers $DATA_NUM_WORKERS
)

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
FFN_HIDDEN_SIZE=8192

MAX_POSITION_EMBEDDINGS=${SEQ_LENGTH}

GPT_MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
)

TP=1
PP=1

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP 
	--pipeline-model-parallel-size $PP
    --sequence-parallel
)

NUM_GPUS=$((SLURM_GPUS_PER_NODE*SLURM_JOB_NUM_NODES))

echo "SLURM_GPUS_PER_NODE: " $SLURM_GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES: " $SLURM_JOB_NUM_NODES 
echo "NUM_GPUS: " $NUM_GPUS
GLOBAL_BATCH_SIZE=$(((NUM_GPUS*MICRO_BATCH_SIZE*GAS)/TP))
TOKENS_GLOBAL_BATCH_SIZE=$((SEQ_LENGTH*GLOBAL_BATCH_SIZE))

echo "MICRO_BATCH_SIZE: " $MICRO_BATCH_SIZE
echo "GRADIENT_ACCUMULATION_STEPS: " $GAS
echo "GLOBAL_BATCH_SIZE: " $GLOBAL_BATCH_SIZE
echo "SEQUENCE LENGTH: " ${SEQ_LENGTH}
echo "TOKENS_GLOBAL_BATCH_SIZE: " ${TOKENS_GLOBAL_BATCH_SIZE}

CHECKPOINT_FORMAT="torch_dist"

if (( TP > 1 || PP > 1 )); then 

    CHECKPOINT_FORMAT="torch_dist"

fi

TOTAL_TOKENS_NUM=${TOKENS_TOTAL} # 300B, 50B tokens
BILLION_NUM=1000000000 # 1B
TOKENS_BILLION=$(( TOTAL_TOKENS_NUM / BILLION_NUM ))
TOTAL_TOKENS_LABEL="${TOKENS_BILLION}B"

COOLDOWN_FRACTION=1/5
TRAIN_ITERS=$(((${TOTAL_TOKENS_NUM} + (${SEQ_LENGTH} * ${GLOBAL_BATCH_SIZE}) - 1)/(${SEQ_LENGTH}*${GLOBAL_BATCH_SIZE})))
LR_DECAY_ITERS=$TRAIN_ITERS
LR_WSD_DECAY_ITERS=$((${TRAIN_ITERS} * ${COOLDOWN_FRACTION}))

SAVE_INTERVAL=${SAVE_INTERVAL_ITERS}
EVAL_INTERVAL=${TRAIN_ITERS}
LOG_INTERVAL=50
EVAL_ITERS=1

echo "TOTAL TOKENS: " $TOTAL_TOKENS_NUM
echo "TOTAL TOKENS LABEL: " $TOTAL_TOKENS_LABEL
echo "TRAIN_ITERS: " $TRAIN_ITERS
echo "LR_WARMUP_ITERS: " $LR_WARMUP_ITERS
echo "LR_DECAY_ITERS: " $LR_DECAY_ITERS
echo "LR_WSD_DECAY_ITERS: " $LR_WSD_DECAY_ITERS

LR_DECAY_STYLE=${LR_SCHEDULE}
LR_WSD_DECAY_STYLE="linear"

echo "LR_WARMUP_ITERS: " $LR_WARMUP_ITERS
echo "LR: " $LR

ROTARY_PERCENT=1.0

NORM_EPSILON=1e-5
INIT_METHOD_STD=0.02

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.05 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --lr-decay-style ${LR_DECAY_STYLE}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr ${LR}
    --min-lr ${LR_MIN}' >> ${template_script_path}/exp_${name_template}.sbatch

if [ "${lr_schedule}" = "WSD" ]
then
echo '    --lr-wsd-decay-style ${LR_WSD_DECAY_STYLE}
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS}' >> ${template_script_path}/exp_${name_template}.sbatch
fi

echo '    --data-cache-path $MEGATRON_CACHE
    --use-flash-attn
    --bf16
    --qk-layernorm  
    --tensorboard-dir $TENSORBOARD_DIR
    --ckpt-format $CHECKPOINT_FORMAT
    --position-embedding-type rope
    --rotary-base ${ROTARY_BASE}
    --rotary-percent ${ROTARY_PERCENT}
    --normalization RMSNorm
    --norm-epsilon ${NORM_EPSILON}
    --init-method-std ${INIT_METHOD_STD}
    --swiglu
    --distributed-backend nccl 
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    --recompute-activations
)


CHECKPOINT_PATH="${RUN_DIR}/checkpoints"
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
EXP_LABEL="open-sci-ref_model-1.7b_data-${DATA_NAME}_samples-${TOTAL_TOKENS_LABEL}_global_bs-${GLOBAL_BATCH_SIZE}_context-${SEQ_LENGTH}_rotary-${ROTARY_BASE}_schedule-${LR_DECAY_STYLE}_lr-${LR}_warmup-${LR_WARMUP_ITERS}"

CHECKPOINT_PATH="$CHECKPOINT_PATH/${EXP_LABEL}"

mkdir -p $CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="$CHECKPOINT_PATH/tensorboard"
mkdir -p $TENSORBOARD_LOGS_PATH


EVAL_AND_LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --save-interval ${SAVE_INTERVAL} 
    --eval-interval ${EVAL_INTERVAL} 
    --log-throughput
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters ${EVAL_ITERS}
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

CMD="pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    "

DISTRIBUTED_ARGS=(
    --nproc-per-node $GPUS_PER_NODE 
    --nnodes $NNODES
)


LAUNCHER="singularity exec \
    --nv \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $MEGATRON_CACHE_FOLDER:$MEGATRON_CACHE_FOLDER \
    $IMAGE \
   python -u -m torch.distributed.run \
    ${DISTRIBUTED_ARGS[@]} \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend static \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD


SRUN_ARGS=" \
    --wait=60 --cpus-per-task=32 --threads-per-core=1 \
    --kill-on-bad-exit=1"

cd $MEGATRON_PATH

srun $SRUN_ARGS \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" &

wait

' >> ${template_script_path}/exp_${name_template}.sbatch


echo "DEBUG: executing sbatch ${template_script_path}/exp_${name_template}.sbatch $micro_bs $gas $lr_schedule $lr $lr_min $lr_warmup_iters $tokens $seq_length $rotary_base $save_interval_iter $data_name $data_switch"
echo "DEBUG: data chunks index to compose: 0 - ${data_chunks}" 

sbatch ${template_script_path}/exp_${name_template}.sbatch $micro_bs $gas $lr_schedule $lr $lr_min $lr_warmup_iters $tokens $seq_length $rotary_base $save_interval_iter $data_name $data_switch