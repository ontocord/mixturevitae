import json
from pathlib import Path
import pandas as pd


metrics = [
    "mmlu/acc",
    # "mmlu_pro/exact_match,custom-extract",
    "copa/acc",
    "lambada_openai/acc",
    # "openbookqa/acc",
    "openbookqa/acc_norm",
    "winogrande/acc",
    "arc_challenge/acc_norm",
    # "arc_challenge/acc",
    "arc_easy/acc_norm",
    # "arc_easy/acc",
    "boolq/acc",
    "commonsense_qa/acc",
    "hellaswag/acc_norm",
    "piqa/acc_norm",
    # "piqa/acc",
    # "social_iqa/acc",
]

bench_sel = [x.split("/")[0] for x in metrics]

hp_cols = [
    'size',
    'dataset',
    'tokenizer',
    'n_tokens',
    'global_batch_size',
    'seq_length',
    'lr_decay_style',
    'lr',
    'lr_warmup_iters']


def sanitize(x):
    return x
    if x.endswith("run-1"):
        x = x[:-5]
    if "machine-LEONARDO":
        res = x.split("machine-LEONARDO")[0]
    # res = x if x.endswith("LEONARDO") else "_".join(x.split("_")[:-1])
    res = res.replace("_", "")
    res = res.lower()
    assert len(res) > 0
    return res


def _add_hp(df, mapping_file):
    if not Path(mapping_file).exists():
        root = Path(__file__).parent
        mapping_path = root / "data" / mapping_file
    else:
        mapping_path = mapping_file
    
    with open(mapping_path, "r") as f:
        mapping_rows = []
        for line in f:
            mapping_rows.append(json.loads(line))
    mapping = {str(Path(row["log_file_name"]).stem): row for row in mapping_rows}

    df_hp = pd.DataFrame([x["config"] for x in df.apply(lambda row: mapping[row["model_name"]], axis=1).tolist()])

    df = pd.concat([df, df_hp], axis=1)
    return df


def format_large_number(num: float):
    """
    Returns:
        str: Formatted string representation (e.g., "50B", "1T")
    """
    if num >= 1e12:
        return f"{int(num / 1e12)}T"
    elif num >= 1e9:
        return f"{int(num / 1e9)}B"
    else:
        return str(num)  # For smaller numbers, just return as is


def load_data(filename: str, mapping_file: str = "log_dir_name_mapping.jsonl"):
    root = Path(__file__).parent
    
    if not Path(filename).exists():
        file_path = root / "data" / filename
        if not file_path.exists():
            # Try with .csv extension
            file_path = root / "data" / f"{filename}.csv"
            if not file_path.exists():
                # Try with .csv.zip extension
                file_path = root / "data" / f"{filename}.csv.zip"
                if not file_path.exists():
                    raise FileNotFoundError(f"Could not find data file: {filename}")
    else:
        file_path = filename
    
    df = pd.read_csv(file_path)
    df = _add_hp(df, mapping_file)
    df["n_iter"] = df.model_path.apply(lambda x: int(x.split("_")[-1]))
    df["metric_name"] = df.apply(lambda row: row["benchmark"] + "/" + row["metric"], axis=1)
    df = df[df.metric_name.isin(metrics)]

    df["n_tokens"] = df.apply(lambda row: format_large_number(row["seq_length"] * row["global_batch_size"] * row["train_iters"]), axis=1)

    def _size(hidden_size, num_layers, ffn_hidden_size, **kwargs):
        model_size = (
            (50432 * hidden_size)
            + (
                num_layers
                * (
                    (4 * hidden_size**2)
                    + 2 * hidden_size * ffn_hidden_size
                    + ffn_hidden_size * hidden_size
                )
            )
        ) / 1_000_000_000

        if model_size < 1.0:
            return float(
                f"{model_size:.2f}"
            )
        else:
            return float(
                f"{model_size:.1f}"
            )

    df["size_calculated"] = df.apply(lambda row: _size(**row), axis=1)
    df["size"] = df.apply(lambda row: float(row["model_size"].replace('b', '')), axis=1)
    return df


def figure_path() -> Path:
    path = Path(__file__).parent / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    load_data("80bt")