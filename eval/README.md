# Evaluation

These are the scripts we used to evaluate MixtureVitae checkpoints. Everything
runs through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
and is submitted as a SLURM job array via [slurmpilot](https://github.com/geoalgo/slurmpilot).

The benchmarks and few-shot settings live in [`tasks.txt`](tasks.txt):

| few-shot | tasks |
|---|---|
| 0  | copa, openbookqa, lambada_openai, winogrande, social_iqa |
| 5  | mmlu_continuation, mmlu |
| 10 | commonsense_qa, piqa, arc_challenge, arc_easy, hellaswag, boolq |

## Before you start

A couple of things worth knowing up front:

- The compute nodes on most HPC clusters have **no internet access**, so the
  environment and all datasets have to be prepared on the login node first. The
  jobs run with `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline`.
- Checkpoints must already be in Hugging Face format. If yours are still in
  Megatron distributed format, convert them first — see `../train/README.md`.

## 1. Set up the environment and download the datasets

Pick a location for the virtual env and the HF cache, then run `setup_node.sh`.
This creates the venv, installs lm-evaluation-harness + accelerate, and pulls
every dataset locally by running a tiny model over all the tasks once.

```bash
export VENV_DIR=~/llmeval/venv      # where the virtual env goes
export HF_HOME=~/llmeval/hf         # where datasets get cached

source setup_node.sh "$VENV_DIR" "$HF_HOME"
```

Run this once on the login node (it needs internet). Re-running it just
re-activates the existing venv.

## 2. Install slurmpilot

The launcher uses slurmpilot to build and submit the job array:

```bash
pip install "slurmpilot[extra] @ git+https://github.com/geoalgo/slurmpilot.git"
```

## 3. List the checkpoints to evaluate

Put the absolute path of each checkpoint on its own line:

```bash
cat > model_to_eval.txt <<'EOF'
/path/to/converted_checkpoints/run-a/hf
/path/to/converted_checkpoints/run-b/hf
EOF
```

## 4. Launch

```bash
python launch_eval.py \
    --model model_to_eval.txt \
    --cluster <cluster_name> \
    --partition <partition> \
    --account <account> \
    --hf_home "$HF_HOME" \
    --venv_path "$VENV_DIR" \
    --eval_output_path evals/
```

This fans out one SLURM job per task group; each job walks through every
checkpoint in `model_to_eval.txt` and runs the eval. Result JSONs land under the
path you passed to `--eval_output_path`.

A few flags that come in handy:

- `--tasks_file mytasks.txt` — use a different task list (defaults to `tasks.txt`).
- `--max_jobs N` / `--start N` — only submit a slice of the jobs, useful for a
  quick smoke test before launching everything.

## 5. Get the results

Metrics are logged to Weights & Biases, but since the nodes are offline you have
to sync the runs afterwards from a machine with internet:

```bash
wandb sync --sync-all
```

The raw per-task JSON files are also written under `--eval_output_path` if you'd
rather aggregate them yourself.
