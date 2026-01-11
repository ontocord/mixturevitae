import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from figure_utils import load_data, metrics, figure_path

# Benchmark name mapping for display
benchmark_display_names = {
    "mmlu": "MMLU",
    "piqa": "PIQA",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC Challenge",
    "commonsense_qa": "CommonsenseQA",
    "arc_easy": "ARC Easy",
    "boolq": "BoolQ",
    "copa": "COPA",
    "lambada_openai": "LAMBADA",
    "openbookqa": "OpenBookQA",
    "winogrande": "Winogrande",
}

# Dataset styling configuration matching figure_for_paper.py style
datasets = {
    'Nemotron-cc-2024-HQ-real-synth-mix': {
        'name': 'Nemotron CC HQ', 'color': '#FF7F0E',  # Orange
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
        'name': 'FineWeb Edu', 'color': '#E377C2',  # Pink
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 6
    },
    'HPLT-2.0': {
        'name': 'HPLT 2.0', 'color': '#9467BD',  # Purple
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 5
    },
    'SlimPajama': {
        'name': 'SlimPajama', 'color': '#7F7F7F',  # Gray
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 4
    },
    'Comma-0.1': {
        'name': 'Comma 0.1', 'color': '#BCBD22',  # Olive
        'linestyle': (0, (1, 1)), 'lw': 2, 'zorder': 3
    },
    'CommonCorpus': {
        'name': 'CommonCorpus', 'color': '#8C564B',  # Brown
        'linestyle': (0, (3, 2)), 'lw': 2, 'zorder': 2
    },
    'CommonCorpus-eng': {
        'name': 'CommonCorpus', 'color': '#8C564B',  # Brown
        'linestyle': (0, (3, 2)), 'lw': 2, 'zorder': 2
    },
    'C4': {
        'name': 'C4', 'color': '#17BECF',  # Cyan
        'linestyle': (0, (3, 1, 1, 1)), 'lw': 2, 'zorder': 1
    },
    'Pile': {
        'name': 'Pile', 'color': '#D62728',  # Red
        'linestyle': (0, (1, 2)), 'lw': 2, 'zorder': 1
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


def plot_single_task(
    data_file="results-08-09.csv.zip",
    mapping_file="log_dir_name_mapping.jsonl",
    model_size='1.7b',
    n_tokens='300B',
    seq_length=4096,
    lr_warmup_iters=25000,
    task="mmlu",
    output_name="dataset_scaling_per_task_paper",
    xlim_max=4.0,
    figsize=(10, 8),
):
    """Plot a single task's performance across datasets with paper styling."""
    df_plot = data_load_filtering(
        data_file, mapping_file, model_size, n_tokens, seq_length, lr_warmup_iters
    )

    print("Available datasets:", df_plot.dataset.unique())

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Filter for the specific task
    df_sub = df_plot.loc[df_plot.benchmark == task, :].copy()

    df_iter_pivot = df_sub.pivot_table(
        index="tokens", columns="dataset", values="value"
    )
    # Don't drop all NA - keep data for each dataset independently

    # Determine label positions
    final_values = {}
    for col in df_iter_pivot.columns:
        if col in datasets:
            col_data = df_iter_pivot[col].dropna()
            if len(col_data) > 0:
                final_values[col] = col_data.values[-1]

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
    # Calculate label positions with spacing
    label_positions = {}
    min_gap = 0.015

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
        x_position = col_data.index[-1] + 0.05 * 1e11
        y_position = label_positions.get(col, col_data.values[-1])
        font_weight = "bold" if col == "MixtureVitae-300BT" else "normal"

        ax.text(
            x_position,
            y_position,
            datasets[col]['name'],
            color=datasets[col]['color'],
            fontsize=14,
            fontweight=font_weight,
            ha='left',  # Align text to the left so it extends to the right
            va='center'
        )

    task_display = benchmark_display_names.get(task, task)
    ax.grid(True, linestyle=':', which='both', color='#aaa', alpha=0.7)
    ax.set_xlabel(f"Number of tokens (100 billions)", fontsize=24)
    ax.set_ylabel(f"{task_display} performance", fontsize=24)

    ax.set_xlim(0, xlim_max * 1e11)
    tick_positions = [i * 0.5 * 1e11 for i in range(int(xlim_max * 2) + 1)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{i * 0.5:.1f}" for i in range(int(xlim_max * 2) + 1)])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    plt.savefig(figure_path() / f"{output_name}_{task}.pdf", bbox_inches='tight', dpi=900)
    plt.savefig(figure_path() / f"{output_name}_{task}.png", bbox_inches='tight', dpi=300)
    print(f"Saved to {figure_path() / output_name}_{task}.pdf and .png")
    plt.show()


def plot_multi_task_grid(
    data_file="results-08-09.csv.zip",
    mapping_file="log_dir_name_mapping.jsonl",
    model_size='1.7b',
    n_tokens='300B',
    seq_length=4096,
    lr_warmup_iters=25000,
    output_name="dataset_scaling_per_task_paper_grid",
    xlim_max=4.0,
    figsize=(20, 12),
    metrics_to_plot=None,
):
    """Plot multiple tasks in a grid layout with paper styling."""
    df_plot = data_load_filtering(
        data_file, mapping_file, model_size, n_tokens, seq_length, lr_warmup_iters
    )

    print("Available datasets:", df_plot.dataset.unique())

    # Use provided metrics or default
    if metrics_to_plot is None:
        metrics_to_plot = metrics.copy()

    n_metrics = len(metrics_to_plot)
    n_cols = 5
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.ravel(axes)

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        bench = metric.split("/")[0]

        df_sub = df_plot.loc[df_plot.metric_name == metric, :].copy()
        df_iter_pivot = df_sub.pivot_table(
            index="tokens", columns="dataset", values="value"
        )
        # Don't drop all NA - keep data for each dataset independently

        if df_iter_pivot.empty:
            ax.set_visible(False)
            continue

        # Plot each dataset
        for col in df_iter_pivot.columns:
            if col not in datasets:
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
                linewidth=datasets[col]['lw'] * 0.7,  # Slightly thinner for grid
                zorder=datasets[col]['zorder']
            )

        task_display = benchmark_display_names.get(bench, bench)
        ax.grid(True, linestyle=':', which='both', color='#aaa', alpha=0.7)
        ax.set_title(task_display, fontsize=14, fontweight='bold')

        ax.set_xlim(0, xlim_max * 1e11)
        tick_positions = [i * 1.0 * 1e11 for i in range(int(xlim_max) + 1)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{i:.0f}" for i in range(int(xlim_max) + 1)])
        ax.tick_params(axis='both', which='major', labelsize=10)

        if idx % n_cols == 0:
            ax.set_ylabel("Performance", fontsize=12)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Tokens (100B)", fontsize=12)

    # Hide unused axes
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)

    # Create legend
    handles = []
    labels = []
    for ds_key, ds_info in datasets.items():
        line, = plt.plot([], [], color=ds_info['color'], linestyle=ds_info['linestyle'],
                         linewidth=ds_info['lw'] * 0.7, label=ds_info['name'])
        handles.append(line)
        labels.append(ds_info['name'])

    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(figure_path() / f"{output_name}.pdf", bbox_inches='tight', dpi=900)
    plt.savefig(figure_path() / f"{output_name}.png", bbox_inches='tight', dpi=300)
    print(f"Saved to {figure_path() / output_name}.pdf and .png")
    plt.show()


def plot_side_by_side(
    data_file="results-08-09.csv.zip",
    mapping_file="log_dir_name_mapping.jsonl",
    model_size='1.7b',
    n_tokens='300B',
    seq_length=4096,
    lr_warmup_iters=25000,
    task="mmlu",
    output_name="dataset_scaling_combined_paper",
    xlim_max=4.0,
    figsize=(16, 6),
):
    """Plot average performance and single task side by side, matching the paper image."""
    df_plot = data_load_filtering(
        data_file, mapping_file, model_size, n_tokens, seq_length, lr_warmup_iters
    )

    print("Available datasets:", df_plot.dataset.unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bench_sel = list(benchmark_display_names.keys())
    print("Selected benchmarks for average plot:", bench_sel)
    print("len(bench_sel):", len(bench_sel))
    # --- Left plot: Average performance ---
    df_sub = df_plot.loc[df_plot.benchmark.isin(bench_sel), :].copy()
    df_iter = df_sub.pivot_table(
        index=["dataset", "tokens"], columns="benchmark", values="value"
    ).loc[:, bench_sel].mean(axis=1)

    df_iter_pivot = df_iter.reset_index().pivot_table(
        index="tokens", columns="dataset", values=0
    )
    # Don't drop all NA - keep data for each dataset independently

    final_values_avg = {}
    for col in df_iter_pivot.columns:
        if col in datasets:
            col_data = df_iter_pivot[col].dropna()
            if len(col_data) > 0:
                final_values_avg[col] = col_data.values[-1]

    sorted_datasets_avg = sorted(final_values_avg.items(), key=lambda x: x[1], reverse=True)

    for col in df_iter_pivot.columns:
        if col not in datasets:
            continue
        col_data = df_iter_pivot[col].dropna()
        if len(col_data) == 0:
            continue
        ax1.plot(
            col_data.index,
            col_data.values,
            label=datasets[col]['name'],
            color=datasets[col]['color'],
            linestyle=datasets[col]['linestyle'],
            linewidth=datasets[col]['lw'],
            zorder=datasets[col]['zorder']
        )

    # Labels for left plot (to the right of line endpoints)
    label_positions_avg = {}
    min_gap = 0.012

    for i, (col, y_val) in enumerate(sorted_datasets_avg):
        if i == 0:
            label_positions_avg[col] = y_val
        else:
            prev_col = sorted_datasets_avg[i-1][0]
            prev_pos = label_positions_avg[prev_col]
            if prev_pos - y_val < min_gap:
                label_positions_avg[col] = prev_pos - min_gap
            else:
                label_positions_avg[col] = y_val

    for col in df_iter_pivot.columns:
        if col not in datasets:
            continue
        col_data = df_iter_pivot[col].dropna()
        if len(col_data) == 0:
            continue
        x_position = col_data.index[-1] + 0.05 * 1e11
        y_position = label_positions_avg.get(col, col_data.values[-1])
        font_weight = "bold" if col == "MixtureVitae-300BT" else "normal"
        ax1.text(
            x_position, y_position, datasets[col]['name'],
            color=datasets[col]['color'], fontsize=11,
            fontweight=font_weight, ha='left', va='center'
        )

    ax1.grid(True, linestyle=':', which='both', color='#aaa', alpha=0.7)
    ax1.set_xlabel("Number of tokens (100 billions)", fontsize=16)
    ax1.set_ylabel("Average downstream performance", fontsize=16)
    ax1.set_xlim(0, xlim_max * 1e11)
    tick_positions = [i * 0.5 * 1e11 for i in range(int(xlim_max * 2) + 1)]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f"{i * 0.5:.1f}" for i in range(int(xlim_max * 2) + 1)])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    # --- Right plot: Single task ---
    df_sub_task = df_plot.loc[df_plot.benchmark == task, :].copy()
    df_iter_pivot_task = df_sub_task.pivot_table(
        index="tokens", columns="dataset", values="value"
    )
    # Don't drop all NA - keep data for each dataset independently

    final_values_task = {}
    for col in df_iter_pivot_task.columns:
        if col in datasets:
            col_data = df_iter_pivot_task[col].dropna()
            if len(col_data) > 0:
                final_values_task[col] = col_data.values[-1]

    sorted_datasets_task = sorted(final_values_task.items(), key=lambda x: x[1], reverse=True)

    for col in df_iter_pivot_task.columns:
        if col not in datasets:
            continue
        col_data = df_iter_pivot_task[col].dropna()
        if len(col_data) == 0:
            continue
        ax2.plot(
            col_data.index,
            col_data.values,
            label=datasets[col]['name'],
            color=datasets[col]['color'],
            linestyle=datasets[col]['linestyle'],
            linewidth=datasets[col]['lw'],
            zorder=datasets[col]['zorder']
        )

    # Labels for right plot (to the right of line endpoints)
    if len(final_values_task) > 0:
        label_positions_task = {}
        min_gap_task = 0.015

        for i, (col, y_val) in enumerate(sorted_datasets_task):
            if i == 0:
                label_positions_task[col] = y_val
            else:
                prev_col = sorted_datasets_task[i-1][0]
                prev_pos = label_positions_task[prev_col]
                if prev_pos - y_val < min_gap_task:
                    label_positions_task[col] = prev_pos - min_gap_task
                else:
                    label_positions_task[col] = y_val

        for col in df_iter_pivot_task.columns:
            if col not in datasets:
                continue
            col_data = df_iter_pivot_task[col].dropna()
            if len(col_data) == 0:
                continue
            x_position_task = col_data.index[-1] + 0.05 * 1e11
            y_position = label_positions_task.get(col, col_data.values[-1])
            font_weight = "bold" if col == "MixtureVitae-300BT" else "normal"
            ax2.text(
                x_position_task, y_position, datasets[col]['name'],
                color=datasets[col]['color'], fontsize=11,
                fontweight=font_weight, ha='left', va='center'
            )

    task_display = benchmark_display_names.get(task, task)
    ax2.grid(True, linestyle=':', which='both', color='#aaa', alpha=0.7)
    ax2.set_xlabel("Number of tokens (100 billions)", fontsize=16)
    ax2.set_ylabel(f"{task_display} performance", fontsize=16)
    ax2.set_xlim(0, xlim_max * 1e11)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{i * 0.5:.1f}" for i in range(int(xlim_max * 2) + 1)])
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    plt.tight_layout()

    plt.savefig(figure_path() / f"{output_name}.pdf", bbox_inches='tight', dpi=900)
    plt.savefig(figure_path() / f"{output_name}.png", bbox_inches='tight', dpi=300)
    print(f"Saved to {figure_path() / output_name}.pdf and .png")
    plt.show()


if __name__ == '__main__':
    # # Plot single MMLU task
    # plot_single_task(
    #     data_file="results-08-09.csv.zip",
    #     mapping_file="log_dir_name_mapping.jsonl",
    #     model_size='1.7b',
    #     n_tokens='300B',
    #     seq_length=4096,
    #     lr_warmup_iters=25000,
    #     task="mmlu",
    #     output_name="dataset_scaling_per_task_paper",
    #     xlim_max=4.0,
    # )

    # Plot side-by-side (average + single task) like in the paper image
    plot_side_by_side(
        data_file="results-08-09.csv.zip",
        mapping_file="log_dir_name_mapping.jsonl",
        model_size='1.7b',
        n_tokens='300B',
        seq_length=4096,
        lr_warmup_iters=25000,
        task="mmlu",
        output_name="dataset_scaling_combined_paper",
        xlim_max=4.0,
    )
