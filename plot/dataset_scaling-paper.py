import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from figure_utils import load_data, bench_sel, figure_path

# Benchmark name mapping
mapping = {
    "mmlu": "MMLU",
    "piqa": "PIQA",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC_Challenge",
    "commonsense_qa": "CommonsenseQA",
    "arc_easy": "ARC_Easy",
    "boolq": "BoolQ",
    "copa": "COPA",
    "lambada_openai": "LAMBADA",
    "openbookqa": "OpenBookQA",
    "winogrande": "Winogrande",
}

bench_sel = mapping.keys()

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

    # Normalize dataset names
    df_plot['dataset_original'] = df_plot['dataset'].copy()

    df_plot = df_plot.loc[
        (df_plot.model_size == model_size) &
        (df_plot.n_tokens == n_tokens) &
        (df_plot.seq_length == seq_length) &
        (df_plot.lr_warmup_iters == lr_warmup_iters)
    ]
    df_plot["tokens"] = df_plot["n_iter"] * df_plot['global_batch_size'] * df_plot['seq_length']
    return df_plot


def plot_dataset_scaling(
    data_file="results-08-09.csv.zip",
    mapping_file="log_dir_name_mapping.jsonl",
    model_size='1.7b',
    n_tokens='300B',
    seq_length=4096,
    lr_warmup_iters=25000,
    output_name="dataset_scaling_paper",
    xlim_max=4.0,
    figsize=(10, 8),
):
    df_plot = data_load_filtering(
        data_file, mapping_file, model_size, n_tokens, seq_length, lr_warmup_iters
    )

    print("Available datasets:", df_plot.dataset.unique())
    print("df_plot.benchmark.value_counts():", df_plot.benchmark.value_counts())

    size = float(model_size.replace('b', ''))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    df_sub = df_plot.loc[df_plot.benchmark.isin(bench_sel), :].copy()
    df_iter = df_sub.pivot_table(
        index=["dataset", "tokens"], columns="benchmark", values="value"
    ).loc[:, bench_sel].mean(axis=1)

    df_iter_pivot = df_iter.reset_index().pivot_table(
        index="tokens", columns="dataset", values=0
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
    min_gap = 0.0075  # Minimum gap between labels (smaller for compact layout)

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
            fontsize=12,  # Smaller font for compact layout
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


def create_final_scores_table(
    data_file="results-08-09.csv.zip",
    mapping_file="log_dir_name_mapping.jsonl",
    model_size='1.7b',
    n_tokens='300B',
    seq_length=4096,
    lr_warmup_iters=25000,
    output_name="final_scores_table",
):
    """
    Create a table with final iteration scores for 1.7B models trained on selected datasets.

    Rows: benchmark names
    Columns: MixtureVitae, Comma-0.1, CommonCorpus, FineWeb-Edu, DCLM
    """
    df_plot = data_load_filtering(
        data_file, mapping_file, model_size, n_tokens, seq_length, lr_warmup_iters
    )

    # Datasets to include in the table (in order)
    table_datasets = [
        'MixtureVitae-300BT',
        'Comma-0.1',
        'CommonCorpus-eng',
        'FineWeb-Edu-1.4T',
        'DCLM',
    ]

    # Display names for columns
    dataset_display_names = {
        'MixtureVitae-300BT': 'MixtureVitae',
        'Comma-0.1': 'Comma-0.1',
        'CommonCorpus-eng': 'CommonCorpus (eng)',
        'FineWeb-Edu-1.4T': 'FineWeb-Edu',
        'DCLM': 'DCLM',
    }

    # Filter for selected benchmarks
    df_sub = df_plot.loc[df_plot.benchmark.isin(bench_sel), :].copy()

    # Get the final (max) iteration for each dataset-benchmark combination
    final_scores = []
    for dataset in table_datasets:
        df_dataset = df_sub[df_sub.dataset == dataset]
        if len(df_dataset) == 0:
            print(f"Warning: No data for dataset {dataset}")
            continue

        # Get max iteration for this dataset
        max_iter = df_dataset.n_iter.max()
        df_final = df_dataset[df_dataset.n_iter == max_iter]

        for bench in bench_sel:
            df_bench = df_final[df_final.benchmark == bench]
            if len(df_bench) > 0:
                score = df_bench['value'].values[0]
                final_scores.append({
                    'Dataset': dataset_display_names[dataset],
                    'Benchmark': mapping.get(bench, bench),
                    'Score': score
                })

    # Create pivot table
    df_scores = pd.DataFrame(final_scores)
    pivot_table = df_scores.pivot_table(
        index='Benchmark',
        columns='Dataset',
        values='Score'
    )

    # Reorder columns to match desired order
    col_order = [dataset_display_names[d] for d in table_datasets if dataset_display_names[d] in pivot_table.columns]
    pivot_table = pivot_table[col_order]

    # Reorder rows to match desired order
    row_order_benchmarks = [
        'copa', 'lambada_openai', 'openbookqa', 'winogrande', 'mmlu',
        'arc_challenge', 'arc_easy', 'boolq', 'commonsense_qa', 'hellaswag', 'piqa'
    ]
    row_order = [mapping.get(b, b) for b in row_order_benchmarks if mapping.get(b, b) in pivot_table.index]
    pivot_table = pivot_table.loc[row_order]

    # Add average row
    avg_row = pivot_table.mean()
    pivot_table.loc['Average'] = avg_row

    # Print table
    print("\n" + "="*80)
    print("Final Iteration Scores for 1.7B Models")
    print("="*80)
    print(pivot_table.to_string(float_format=lambda x: f"{x:.2f}"))

    # Save as CSV
    csv_path = figure_path() / f"{output_name}.csv"
    pivot_table.to_csv(csv_path)
    print(f"\nSaved CSV to {csv_path}")

    # Generate LaTeX table
    latex_table = pivot_table.to_latex(
        float_format=lambda x: f"{x:.2f}",
        caption="Final iteration benchmark scores for 1.7B models trained on different datasets.",
        label="tab:final_scores",
    )

    latex_path = figure_path() / f"{output_name}.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {latex_path}")

    return pivot_table


if __name__ == '__main__':
    plot_dataset_scaling(
        data_file="results-08-09.csv.zip",
        mapping_file="log_dir_name_mapping.jsonl",
        model_size='1.7b',
        n_tokens='300B',
        seq_length=4096,
        lr_warmup_iters=25000,
        output_name="300B_results_all_plot_clustered_avg",
        xlim_max=4.0,
    )

    # Create the final scores table
    create_final_scores_table(
        data_file="results-08-09.csv.zip",
        mapping_file="log_dir_name_mapping.jsonl",
        model_size='1.7b',
        n_tokens='300B',
        seq_length=4096,
        lr_warmup_iters=25000,
        output_name="final_scores_table",
    )
