# MixtureVitee
MixtureVitae: A Permissive, High-Performance, Open-Access Pretraining Dataset

## Overview

This repository contains the code and instructions for training and evaluating models using the MixtureVitae dataset. Our training framework utilizes a fork of NVIDIA's Megatron-LM for large-scale model training.

## Setup

We use the [Megatron-LM fork](https://github.com/timurcarstensen/Megatron-LM-Open-Sci) as our training framework. Please follow the setup instructions from the repository to:
- Create a Python environment
- Install required dependencies

## Tokenization

The tokenization scripts are located in the `tokenize` folder:
- **`preprocess_data_parallel.py`**: Python script for parallel data tokenization
- **`tokenize.sh`**: Shell script used to tokenize all datasets from our experiments

These scripts enable efficient preprocessing of large-scale datasets for model training.

## Training

Training procedures and scripts are available in the `train` folder, including:
- Environment variable configuration
- Shell scripts for running experiments
- Support for multiple model sizes: 0.13B, 0.4B, 1.3B, and 1.7B parameters
- Scripts to replicate all the experimental results reported in the paper

**Note**: The training scripts are designed for HPC systems with SLURM job scheduling. Checkpoints are saved in PyTorch distributed format and require conversion to Hugging Face format for evaluation (see Evaluation section). We provide the conversion script and the process to run in the `train/README.md`

## Evaluation

Evaluation scripts are provided in the `eval` folder:
- We use the `lm-evaluation-harness` framework for all benchmarks
- Automated scripts for efficient SLURM job submission on HPC systems

## Decontamination
The decontamination scripts are provided in the `decontamination` folder.

## Math Word Problems
The script for generating math word problems is in the `math_word_problems` folder.

## Data Curation
The script containing the data curation pipeline that processes raw text data from multiple sources (Common-Pile, curated datasets, FineFine, Nemo, MAGA, txt360) is in `process_data` folder.