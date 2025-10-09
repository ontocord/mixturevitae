import argparse
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from pathlib import Path

from figure_utils import load_data, bench_sel, figure_path

colors = [
    "#1F77B4",  # Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#D62728",  # Red
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#7F7F7F",  # Gray
    "#BCBD22",  # Olive
    "#17BECF",  # Cyan
    "#000000",  # Black
    "#3366FF",  # Light Blue
    "#6666FF",  # Light Blue
    "#FF66FF",  # Light Purple
    "#FF9933",  # Light Orange
]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


def plot_avg(
    data_file,
    mapping_file,
    n_tokens,
    sizes,
    output_name,
    seq_length=4096,
    lr_warmup_iters=1000,
    xlim_max=1e11,
    figsize=(11, 4),
    output_dir=None,
):
    """Plot ablation study across different model sizes."""
    
    df_all = load_data(data_file, mapping_file)
    
    fig, axes = plt.subplots(1, len(sizes), figsize=figsize, sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    
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
    
    for i, (ax, size) in enumerate(zip(axes, sizes)):
        df_sub = df_plot.loc[(mask) & (df_plot.loc[:, "size"] == size)].copy()
        df_sub["tokens"] = df_sub["n_iter"] * df_sub["global_batch_size"] * df_sub["seq_length"]
        
        df_iter = df_sub.pivot_table(index=["dataset", "tokens"], columns="benchmark", values="value").loc[:, bench_sel].mean(axis=1)
        df_iter_pivot = df_iter.reset_index().pivot_table(index="tokens", columns="dataset", values=0)
        
        dataset_order = [
            "curated",
            "wo_paperbook",
            "wo_codetech",
            "wo_miscurated",
        ]
        dataset_order = [x.lower() for x in dataset_order]
        
        # fix order to have same colors across plots
        df_iter_pivot = df_iter_pivot.loc[:, [x for x in dataset_order if x in df_iter_pivot.columns]]
        df_iter_pivot.columns = [x.capitalize() for x in df_iter_pivot.columns]
        df_iter_pivot = df_iter_pivot.dropna(how="any")  # drop na tokens (for all datasets)
        
        df_iter_pivot.plot(ax=ax)
        ax.grid()
        ax.set_title(f"{size}B")
        ax.set_xlabel("Number of tokens")
        ax.set_ylabel(f"Average downstream performance")
        
        if i == 0:
            ax.legend(loc="lower left", ncols=1)
        else:
            ax.get_legend().remove()
        
        ax.set_xlim(0, xlim_max)
    
    fig.suptitle(f"Average performance while training", y=0.97)
    
    # Create output directory if it doesn't exist
    out_dir = Path(output_dir) if output_dir else figure_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / f"{output_name}.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.tight_layout()
    print(f"Saved figure to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot ablation study for model training')
    parser.add_argument('--data-file', type=str, default='results-100bt',
                        help='Data file name (should be in /data) or path')
    parser.add_argument('--mapping-file', type=str, default='log_dir_name_mapping.jsonl',
                        help='Mapping JSONL file name (should be in /data) or path')
    parser.add_argument('--n-tokens', type=str, default='100B',
                        help='Number of tokens (e.g., 100B, 1T)')
    parser.add_argument('--sizes', type=float, nargs='+', default=[0.13, 0.4, 1.3, 1.7],
                        help='Model sizes to plot (e.g., 0.13 0.4 1.3 1.7)')
    parser.add_argument('--output-name', type=str, default='100B_curated_ablation',
                        help='Output file name (without extension)')
    parser.add_argument('--seq-length', type=int, default=4096,
                        help='Sequence length')
    parser.add_argument('--lr-warmup-iters', type=int, default=1000,
                        help='Learning rate warmup iterations')
    parser.add_argument('--xlim-max', type=float, default=1e11,
                        help='Maximum x-axis limit')
    parser.add_argument('--figsize', type=float, nargs=2, default=[11, 4],
                        help='Figure size (width height)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: figures/)')
    
    args = parser.parse_args()
    
    plot_avg(
        data_file=args.data_file,
        mapping_file=args.mapping_file,
        n_tokens=args.n_tokens,
        sizes=args.sizes,
        output_name=args.output_name,
        seq_length=args.seq_length,
        lr_warmup_iters=args.lr_warmup_iters,
        xlim_max=args.xlim_max,
        figsize=tuple(args.figsize),
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()