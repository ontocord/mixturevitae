# Multi-Node Distributed Tokenization Pipeline

A distributed data tokenization pipeline using Ray for multi-node parallelization on HPC clusters with SLURM.

## Installation

For installation, we recommend following the installation setups in [MegatronLM](https://github.com/NVIDIA/Megatron-LM)

## Setup

Declare environment the variables.
```bash
export MEGATRON_PATH="mixturevitae/Megatron-LM"
export HF_HOME=<cache-path>

export _INPUT=<input-data>
export _OUTPUT_PREFIX=<output-data>
```

## Tokenize

```bash
sbatch tokenize.sh
```