# Eval Setup

## Environment Setup

Replace the placeholders with your actual paths:

```bash
export HF_HOME="<cache_path>"
export VENV_DIR="<venv_path>"
```

## Eval Command

```bash
python launch_eval.py 
    --model model_to_eval.txt \
    --cluster <cluster_name> \
    --partition <partition> \
    --account <account> \
    --hf_home $HF_HOME \
    --venv_path $VENV_DIR \
    --eval_output_path evals/
```

## Notes

- Replace all placeholders (e.g., `<cluster_name>`, `<cache_path>`, `<partition>`, `<account>`) with actual paths
- Include the absolute paths of all the checkpoints (line separated) to evaluate in `model_to_eval.txt`
