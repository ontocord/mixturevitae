import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from figure_utils import load_data, figure_path

# Dataset styling configuration matching figure_for_paper.py style
datasets = {
    'Nemotron-cc-2024-HQ-real-synth-mix': {
        'name': 'Nemotron-CC-HQ', 'color': '#FF7F0E',  # Orange
        'linestyle': '-', 'lw': 2.5, 'zorder': 9
    },
    'DCLM': {
        'name': 'DCLM', 'color': '#1F77B4',  # Blue
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 7
    },
    'MixtureVitae-300BT': {
        'name': 'MixtureVitae (Ours)', 'color': '#2ca02c',  # Green
        'linestyle': '-', 'lw': 3.5, 'zorder': 10  # Thick and on top
    },
    'FineWeb-Edu-1.4T': {
        'name': 'FineWeb-Edu', 'color': '#E377C2',  # Pink
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 6
    },
    'HPLT-2.0': {
        'name': 'HPLT-2.0', 'color': '#9467BD',  # Purple
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 5
    },
    'SlimPajama': {
        'name': 'SlimPajama', 'color': '#7F7F7F',  # Gray
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 4
    },
    'Comma-0.1': {
        'name': 'Comma-0.1', 'color': '#BCBD22',  # Olive
        'linestyle': (0, (1, 1)), 'lw': 2, 'zorder': 3
    },
    'CommonCorpus-eng': {
        'name': 'CommonCorpus', 'color': '#8C564B',  # Brown
        'linestyle': (0, (3, 2)), 'lw': 2, 'zorder': 2
    },
}


def data_load_filtering(data_file, mapping_file, model_size='1.7b', n_tokens='300B', seq_length=4096, lr_warmup_iters=25000):
    df_all = load_data(data_file, mapping_file)
    df_plot = df_all.copy()

    df_plot = df_plot.loc[
        (df_plot.model_size == model_size) &
        (df_plot.n_tokens == n_tokens) &
        (df_plot.seq_length == seq_length) &
        (df_plot.lr_warmup_iters == lr_warmup_iters)
    ]
    df_plot["tokens"] = df_plot["n_iter"] * df_plot['global_batch_size'] * df_plot['seq_length']
    return df_plot


def plot_dataset_scaling_mmlu(
    data_file="results-08-09.csv.zip",
    mapping_file="log_dir_name_mapping.jsonl",
    model_size='1.7b',
    n_tokens='300B',
    seq_length=4096,
    lr_warmup_iters=25000,
    output_name="dataset_scaling_mmlu_paper",
    xlim_max=4.0,
    figsize=(10, 8),
):
    df_plot = data_load_filtering(
        data_file, mapping_file, model_size, n_tokens, seq_length, lr_warmup_iters
    )

    print("Available datasets:", df_plot.dataset.unique())
    print("df_plot.benchmark.value_counts():", df_plot.benchmark.value_counts())

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Filter for MMLU benchmark only
    df_sub = df_plot.loc[df_plot.benchmark == "mmlu", :].copy()

    df_iter_pivot = df_sub.pivot_table(
        index="tokens", columns="dataset", values="value"
    )
    # Don't drop all NA - keep data for each dataset independently

    # Determine label positions to avoid overlap
    final_values = {}
    for col in df_iter_pivot.columns:
        if col in datasets:
            col_data = df_iter_pivot[col].dropna()
            if len(col_data) > 0:
                final_values[col] = col_data.values[-1]

    # Sort by final value to help with label positioning
    sorted_datasets = sorted(final_values.items(), key=lambda x: x[1], reverse=True)

    # Plot each dataset
    for col in df_iter_pivot.columns:
        if col not in datasets:
            print(f"Skipping unknown dataset: {col}")
            continue

        col_data = df_iter_pivot[col].dropna()
        if len(col_data) == 0:
            continue

        ax.plot(
            col_data.index,
            col_data.values,
            label=datasets[col]['name'],
            color=datasets[col]['color'],
            linestyle=datasets[col]['linestyle'],
            linewidth=datasets[col]['lw'],
            zorder=datasets[col]['zorder']
        )

    # Add labels at the end of lines (to the right of the line endpoints)
    # Calculate label positions with spacing to avoid overlap
    label_positions = {}
    min_gap = 0.006  # Minimum gap between labels (smaller for compact layout)

    for i, (col, y_val) in enumerate(sorted_datasets):
        if i == 0:
            label_positions[col] = y_val
        else:
            prev_col = sorted_datasets[i-1][0]
            prev_pos = label_positions[prev_col]
            if prev_pos - y_val < min_gap:
                label_positions[col] = prev_pos - min_gap
            else:
                label_positions[col] = y_val

    for col in df_iter_pivot.columns:
        if col not in datasets:
            continue

        col_data = df_iter_pivot[col].dropna()
        if len(col_data) == 0:
            continue

        # Position label to the right of the line's last point
        x_position = col_data.index[-1] + 0.02 * 1e11
        y_position = label_positions.get(col, col_data.values[-1])
        font_weight = "bold" if col == "MixtureVitae-300BT" else "normal"

        ax.text(
            x_position,
            y_position,
            datasets[col]['name'],
            color=datasets[col]['color'],
            fontsize=11,  # Smaller font for compact layout
            fontweight=font_weight,
            ha='left',  # Align text to the left so it extends to the right
            va='center'
        )

    ax.grid(True, linestyle=':', which='both', color='#aaa', alpha=0.7)
    ax.set_xlabel(f"Number of tokens (100 billions)", fontsize=24)
    ax.set_ylabel(f"Average downstream performance", fontsize=24)

    # Set x-axis limits and ticks (in units of 100B = 1e11)
    ax.set_xlim(0, xlim_max * 1e11)
    tick_positions = [i * 0.5 * 1e11 for i in range(int(xlim_max * 2) + 1)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{i * 0.5:.1f}" for i in range(int(xlim_max * 2) + 1)])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    plt.savefig(figure_path() / f"{output_name}.pdf", bbox_inches='tight', dpi=900)
    plt.savefig(figure_path() / f"{output_name}.png", bbox_inches='tight', dpi=300)
    print(f"Saved to {figure_path() / output_name}.pdf and .png")
    plt.show()


if __name__ == '__main__':
    plot_dataset_scaling_mmlu(
        data_file="results-08-09.csv.zip",
        mapping_file="log_dir_name_mapping.jsonl",
        model_size='1.7b',
        n_tokens='300B',
        seq_length=4096,
        lr_warmup_iters=25000,
        output_name="300B_results_mmlu_plot_clustered_avg",
        xlim_max=4.0,
    )
