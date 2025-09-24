#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=data_preprocess
#SBATCH --account=<account>
#SBATCH -p boost_usr_prod
#SBATCH --threads-per-core=1
#SBATCH --time=23:59:00
#SBATCH --output=<slurm_output_path>

START_TIME=$(date +%s)

CONDA_ENV="/leonardo_work/AIFAC_L01_028/.cache/envs/hraj0000/dcnlp_py3.10"
MINICONDA_PATH="/leonardo_work/AIFAC_L01_028/.cache/miniconda/miniconda"
source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


# head_node_i="${head_node}i"
head_node_i="${head_node}"

head_node_ip="$(nslookup "$head_node_i" | grep -oP '(?<=Address: ).*')"
echo "Head node: $head_node_ip"


export HF_HOME="/leonardo_work/AIFAC_L01_028/.cache"

export TOKENIZERS_PARALLELISM=false

port=20156
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"



echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node"  \
    $APPTAINER_ARG \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus ${SLURM_CPUS_PER_TASK} --block &

sleep 5

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node=${nodes_array[$i]}
    node_i="${node}i"
    echo "Starting WORKER $i at $node"
    this_node_ip="$(nslookup "$node_i" | grep -oP '(?<=Address: ).*')"
    srun --nodes=1 --ntasks=1 -w "$node" \
              $APPTAINER_ARG \
        ray start --address "$ip_head" \
        --node-ip-address="$this_node_ip"  \
        --num-cpus ${SLURM_CPUS_PER_TASK} --block &
    sleep 10
done

export RAY_ADDRESS="$head_node_ip:$port"


MEGATRON_PATH="/leonardo_work/AIFAC_L01_028/hraj0000/work/megatron/Megatron-LM"
cd $MEGATRON_PATH
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

_INPUT="/leonardo_work/AIFAC_L01_028/shared/datasets/language/tokenized/MixtureVitae-300BT-permissive_notriskmitigated"
_OUTPUT_PREFIX="/leonardo_work/AIFAC_L01_028/shared/datasets/language/tokenized/MixtureVitae-300BT-permissive_notriskmitigated"
if [ -z "$1" ]; then
  INPUT="$_INPUT"
else
  INPUT="$1"
fi
if [ -z "$2" ]; then
  OUTPUT_PREFIX="$_OUTPUT_PREFIX"
else
  OUTPUT_PREFIX="$2"
fi

echo "######### INPUT: $INPUT"
echo "######### OUTPUT_PREFIX: $OUTPUT_PREFIX"

mkdir $OUTPUT_PREFIX
mkdir "${OUTPUT_PREFIX}/GPT-NeoX"

SCRIPT="preprocess_data_parallel.py"
NUM_CPUS_PER_WORKER=4
CPUS_PER_RAY_WORKER=$(($NUM_CPUS_PER_WORKER * 2))
NUM_WORKERS_PER_RAY_PROCESS=$(($SLURM_CPUS_PER_TASK / $CPUS_PER_RAY_WORKER))
CMD="python $SCRIPT \
    --input $INPUT \
    --output-prefix $OUTPUT_PREFIX \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model EleutherAI/gpt-neox-20b \
    --workers $NUM_WORKERS_PER_RAY_PROCESS \
    --cpus-per-ray-worker $CPUS_PER_RAY_WORKER"

$APPTAINER_ARG $CMD