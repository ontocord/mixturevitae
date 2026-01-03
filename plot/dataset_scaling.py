import argparse
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
    lr_warmup_iters=25000,
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
            # "wo_tier2b",
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
        
        # fix order to have same colors across plots
        df_iter_pivot = df_iter_pivot.loc[:, [x for x in dataset_order if x in df_iter_pivot.columns]]
        df_iter_pivot.columns = [x.capitalize() for x in df_iter_pivot.columns]
        df_iter_pivot = df_iter_pivot.dropna(how="any")  # drop na tokens (for all datasets)
        
        df_iter_pivot.plot(ax=ax)
        ax.grid()
        ax.set_title(f"{size}B")
        ax.set_xlabel("Number of tokens")
        ax.set_ylabel("Average downstream performance")
        
        if i == 0:
            ax.legend(loc="lower left", ncols=1)
        else:
            ax.get_legend().remove()
        
        ax.set_xlim(0, xlim_max)
    
    fig.suptitle("Average performance while training", y=0.97)
    
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
    parser.add_argument('--data-file', type=str, default='results-wo_subparts',
                        help='Data file name (should be in /data) or path')
    parser.add_argument('--mapping-file', type=str, default='log_dir_name_mapping-wo_subparts.jsonl',
                        help='Mapping JSONL file name (should be in /data) or path')
    parser.add_argument('--n-tokens', type=str, default='100B',
                        help='Number of tokens (e.g., 100B, 1T)')
    parser.add_argument('--sizes', type=float, nargs='+', default=[0.13, 0.4, 1.3, 1.7],
                        help='Model sizes to plot (e.g., 0.13 0.4 1.3 1.7)')
    parser.add_argument('--output-name', type=str, default='100B_wo_subparts',
                        help='Output file name (without extension)')
    parser.add_argument('--seq-length', type=int, default=4096,
                        help='Sequence length')
    parser.add_argument('--lr-warmup-iters', type=int, default=25000,
                        help='Learning rate warmup iterations')
    parser.add_argument('--xlim-max', type=float, default=3e11,
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
    
# python dataset_scaling.py --data-file results-parts_removal_ablations.csv --mapping-file log_dir_name_mapping-parts_removal_ablations.jsonl --n-tokens 100B --sizes 1.7 --output-name 100B_parts_removal_ablation