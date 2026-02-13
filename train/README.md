# Training Setup

## Environment Setup

### Environment Variables
Replace the placeholders with your actual paths:

```bash
export core_path="<root>/mixturevitae/Megatron-LM"
export HF_HOME="<cache_path>"
export PROJECT_DIR="/leonardo_work/AIFAC_L01_028"
export RUN_DIR="<root>/megatron_lm_reference"
export SHARED_CONTAINERS="<container_path>"
export MEGATRON_CACHE_BASE="<cache_path>"
export DATA="<root>/data"
export TOKENIZER_MODEL="EleutherAI/gpt-neox-20b"
export MEGATRON_PATH="<root>/mixturevitae/Megatron-LM"
```

### SLURM Configuration (if using SLURM)
Edit these in your SLURM script:

```bash
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --output=<slurm_out_path>

IMAGE=${SHARED_CONTAINERS}/pytorch_24.09-py3.sif
```

## Training Commands

The general command template is

```bash
source megatron_exp_data-NAME.sh NODES_NUM TOKENS_NUM MICRO_BS GAS CONTEXT_LENGTH LR_SCHEDULE_TYPE LR END_LR WARMUP_ITERS SAVE_INTERVAL_ITERS DATASET TOKENIZER DATA_CHUNKS
```

where NODES_NUM: number of compute nodes; TOKENS_NUM: tokens for training; MICRO_BS: local batch size per GPU; GAS: gradient accumulation steps; CONTEXT_LENGTH: model context length; LR_SCHEDULE_TYPE: learning rate schedule; LR: peak learning rate; END_LR: end learning rate; WARMUP_ITERS: warm up iterations (steps); SAVE_INTERVAL_ITERS: checkpoint saving interval; DATASET: dataset name;  TOKENIZER: tokenizer name; DATA_CHUNKS: N - 1, where N: number of chunks the dataset is split into

### 50BT Training Runs

#### 1.7B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-1.7b_rotary-PARAM_dist_ckpt_dataIter.sh \
    63 50000000000 4 1 4096 100000 WSD 4e-3 0 1000 1000 <data_dir_name> GPT-NeoX 0
```

#### 1.3B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-1.3b_rotary-PARAM_dist_ckpt_dataIter.sh \
    63 50000000000 4 1 4096 100000 WSD 4e-3 0 1000 1000 <data_dir_name> GPT-NeoX 0
```

#### 0.4B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-0.4b_rotary-PARAM_dist_ckpt_dataIter.sh \
    25 50000000000 10 1 4096 100000 WSD 4e-3 0 1000 1000 <data_dir_name> GPT-NeoX 0
```

#### 0.13B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-0.13b_rotary-PARAM_dist_ckpt_dataIter.sh 
    21 50000000000 12 1 4096 100000 WSD 4e-3 0 1000 1000 <data_dir_name> GPT-NeoX 0
```

### 300BT Training Runs

#### 1.7B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-1.7b_dist_ckpt_dataIter.sh \
    63 300000000000 4 1 4096 WSD 4e-3 0 25000 2000 <data_dir_name> GPT-NeoX 0
```

#### 1.3B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-1.3b_dist_ckpt_dataIter.sh \
    63 300000000000 4 1 4096 WSD 4e-3 0 25000 2000 <data_dir_name> GPT-NeoX 0
```

#### 0.4B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-0.4b_dist_ckpt_dataIter.sh \
    25 300000000000 10 1 4096 WSD 4e-3 0 25000 2000 <data_dir_name> GPT-NeoX 0
```

#### 0.13B Model
```bash
source megatron_exp_data-PARAM_machine-LEONARDO_modelscale-0.13b_dist_ckpt_dataIter.sh \
    21 300000000000 12 1 4096 WSD 4e-3 0 25000 2000 <data_dir_name> GPT-NeoX 0
```

## Notes

- Replace all placeholders (e.g., `<root>`, `<cache_path>`, `<data_dir_name>`) with actual paths
- Ensure all environment variables are set before running training scripts


# MegatronLM to Huggingface Convertor
After training we need to convert the Megatron-LM checkpoint files to HF format for evaluation.
Use the below command for the conversion.

```bash
python consolidated_conversion_workflow.py 
    --slurm_log_dir slurm_output \
    --opensci_megatron_path Megatron-LM \
    --open_sci_hf_path Open-Sci-hf \ 
    --save_checkpoints_dir converted_checkpoints \
    --account <account> \
    --convert_logs_dir slurm_output \
    --checkpoint_paths_and_logs <checkpoint_path>,<log_paths>
```
