import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from figure_utils import load_data, metrics, figure_path


def plot_per_benchmark(
    data_file,
    mapping_file,
    n_tokens,
    size,
    output_name,
    seq_length=4096,
    lr_warmup_iters=1000,
    figsize=(16, 6),
    output_dir=None,
    metric_to_remove="commonsense_qa/acc",
    dataset_order=None,
):
    """Plot per-benchmark performance across different datasets."""
    
    df_all = load_data(data_file, mapping_file)
    
    # Remove metric to get desired subplot layout
    metrics_plot = metrics.copy()
    if metric_to_remove and metric_to_remove in metrics_plot:
        metrics_plot.remove(metric_to_remove)
    
    # Create subplot grid
    n_metrics = len(metrics_plot)
    n_cols = n_metrics // 2
    fig, axes = plt.subplots(2, n_cols, figsize=figsize, sharey=False, sharex=True)
    axes = np.ravel(axes)
    
    df_plot = df_all.copy()
    df_plot.dataset = df_plot.dataset.apply(lambda s: s.replace('Nemotron-cc-2024-HQ-real-synth-mix', 'Nemotron-cc-hq'))
    df_plot.dataset = df_plot.dataset.str.lower()
    
    config = {
        "n_tokens": n_tokens,
        "seq_length": seq_length,
        "lr_warmup_iters": lr_warmup_iters,
    }
    
    mask = None
    for key, value in config.items():
        if mask is None:
            mask = (df_plot.loc[:, key] == value)
        else:
            mask &= (df_plot.loc[:, key] == value)
    
    df_sub = df_plot.loc[(mask) & (df_plot.loc[:, "size"] == size)].copy()
    df_sub["tokens"] = df_sub["n_iter"] * df_sub["global_batch_size"] * df_sub["seq_length"]
    
    # Default dataset order if not provided
    if dataset_order is None:
        dataset_order = [
            # "arxiv",
            # "pes2o",
            # "pubmed",
            # "wiki",
            # "europat",
            # "python_edu_repo",
            # "science_tech",
            # "software",
            # "stackexchange",
            # "all",
            # "wo_paperbook",
            # "wo_codetech",
            # "wo_miscurated"
            # "mixture",
            # "comma",
            # "commoncorpus-eng",
            # "Nemotron-cc-hq",
            # "DCLM",
            # "HPLT-2.0",
            # "FineWeb-Edu-1.4T",
            # "Pile",
            # "SlimPajama",
            # "CommonCorpus",
            # "C4",
            # "all",
            # "wo_europat",
            # "wo_pyedurepo",
            # "wo_sciencetech",
            # "wo_stackexchange",
            # "wo_arxiv",
            # "wo_pes2o"
            "MixtureVitae-300BT-Decontam",
            "MixtureVitae-300BT"
        ]
    dataset_order = [x.lower() for x in dataset_order]
    
    for i, metric in enumerate(metrics_plot):
        ax = axes[i]
        bench = metric.split("/")[0]
        
        df_iter_pivot = df_sub.loc[df_sub.metric_name == metric, :].pivot_table(
            index="tokens", columns="dataset", values="value"
        )
        
        # Fix order to have same colors across plots
        df_iter_pivot = df_iter_pivot.loc[:, [x for x in dataset_order if x in df_iter_pivot.columns]]
        df_iter_pivot.columns = [x.capitalize() for x in df_iter_pivot.columns]
        df_iter_pivot = df_iter_pivot.dropna(how="any")  # drop na tokens (for all datasets)
        
        plot_result = df_iter_pivot.plot(ax=ax)
        ax.grid()
        ax.set_title(f"{bench}")
        ax.set_xlabel("Number of tokens")
        
        if i == 0 or i == n_cols:
            ax.set_ylabel("Downstream performance")
        
        # Store the lines and labels from the first plot
        if i == 0:
            lines = plot_result.get_lines()
            labels = df_iter_pivot.columns.tolist()
        
        ax.get_legend().remove()
    
    # Create a single legend outside the plot
    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.03, 0.5))
    fig.suptitle("Performance while training for different datasets", y=0.97)
    
    # Create output directory if it doesn't exist
    out_dir = Path(output_dir) if output_dir else figure_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / f"{output_name}.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot per-benchmark performance for different datasets')
    parser.add_argument('--data-file', type=str, default='results-wo_subparts',
                        help='Data file name (with or without .csv/.csv.zip extension)')
    parser.add_argument('--mapping-file', type=str, default='log_dir_name_mapping-wo_subparts.jsonl',
                        help='Mapping JSONL file name')
    parser.add_argument('--n-tokens', type=str, default='300B',
                        help='Number of tokens (e.g., 80B, 1T)')
    parser.add_argument('--size', type=float, default=1.7,
                        help='Model size to plot')
    parser.add_argument('--output-name', type=str, default='100B_wo_subparts_per_dataset',
                        help='Output file name (without extension)')
    parser.add_argument('--seq-length', type=int, default=4096,
                        help='Sequence length')
    parser.add_argument('--lr-warmup-iters', type=int, default=25000,
                        help='Learning rate warmup iterations')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 6],
                        help='Figure size (width height)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: figures/)')
    parser.add_argument('--metric-to-remove', type=str, default='commonsense_qa/acc',
                        help='Metric to remove for subplot layout (default: commonsense_qa/acc)')
    parser.add_argument('--dataset-order', type=str, nargs='+', default=None,
                        help='Order of datasets (e.g., curated wo_paperbook wo_codetech)')
    
    args = parser.parse_args()
    
    plot_per_benchmark(
        data_file=args.data_file,
        mapping_file=args.mapping_file,
        n_tokens=args.n_tokens,
        size=args.size,
        output_name=args.output_name,
        seq_length=args.seq_length,
        lr_warmup_iters=args.lr_warmup_iters,
        figsize=tuple(args.figsize),
        output_dir=args.output_dir,
        metric_to_remove=args.metric_to_remove,
        dataset_order=args.dataset_order,
    )


if __name__ == '__main__':
    main()
    
# python dataset_scaling_per_task.py --data-file results-parts_removal_ablations.csv --mapping-file log_dir_name_mapping-parts_removal_ablations.jsonl --n-tokens 100B --size 1.7 --output-name 100B_parts_removal_ablation-per_dataset