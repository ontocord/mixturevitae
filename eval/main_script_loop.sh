set -e

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <TASKS> <NUM_FEWSHOT> <MODEL_PATH_OR_NAME_FILE>"
    exit 1
fi
TASKS=$1           
NUM_FEWSHOT=$2     
MODEL_PATH_OR_NAME_FILE=$3

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

: "${LM_EVAL_OUTPUT_PATH:=evals}"
: "${BATCH_SIZE:=36}"

while IFS= read -r raw_model || [[ -n $raw_model ]]; do
    MODEL_PATH_OR_NAME=$(echo "$raw_model" | xargs)
    [[ -z $MODEL_PATH_OR_NAME ]] && continue

    export OUTLINES_CACHE_DIR=/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/$MODEL_PATH_OR_NAME

    mkdir -p "$LM_EVAL_OUTPUT_PATH"
    OUTPUT_PATH="${LM_EVAL_OUTPUT_PATH%/}${SLURM_ARRAY_JOB_ID}"

    TASK_STR=${TASKS//,/_}
    MODEL_STR=$(echo "$MODEL_PATH_OR_NAME" | sed -E 's#.*/converted_checkpoints/(.+)/hf.*#\1#')
    WANDB_NAME="${SLURM_ARRAY_JOB_ID}-${MODEL_STR}"

    CMD="accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH_OR_NAME,trust_remote_code=True \
    --tasks $TASKS \
    --output_path $OUTPUT_PATH/$MODEL_STR/${TASK_STR}_${NUM_FEWSHOT} \
    --num_fewshot $NUM_FEWSHOT \
    --trust_remote_code \
    --batch_size $BATCH_SIZE \
    --wandb_args project=lm-eval-harness-integration,name=$WANDB_NAME"

    echo "Running command:"
    echo "$CMD"
    echo ""

    $CMD

done < "$MODEL_PATH_OR_NAME_FILE"
