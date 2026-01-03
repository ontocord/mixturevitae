import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from figure_utils import load_data, bench_sel, hp_cols, figure_path

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
# bench_sel = [mapping[i] for i in bench_sel]
# bench_sel = ["MMLU"]

datasets = {
    'MixtureVitae-300BT': {
        'name': 'MixtureVitae', 'group': 'hero', 'color': '#2ca02c', # Solid Green
        'linestyle': '-', 'lw': 3.5, 'zorder': 10 # Thick and on top
    },
    'MixtureVitae-300BT-Decontam': {
        'name': 'MixtureVitae (Decontaminated)', 'group': 'competitor', 'color': '#9467bd', # Purple
        'linestyle': (0, (5, 3)), 'lw': 2, 'zorder': 9
    }
}
name_color_map = {i['name']:i['color'] for key, i in datasets.items()}

def data_load_filtering(model_size = '1.7b', n_tokens = '300B', seq_length = 4096, il_warmup_iters = 25000):
    df_all = load_data("/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/plot/data/results-mxv_decontam.csv", 
                       "/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/plot/data/log_dir_name_mapping-mxv_decontam.jsonl")
    df_plot = df_all.copy()
    df_plot = df_plot.loc[(df_plot.model_size == model_size) & (df_plot.n_tokens == n_tokens) & (df_plot.seq_length == seq_length) & (df_plot.lr_warmup_iters == il_warmup_iters)]
    # df_plot["tokens"] = df_plot["n_iter"] * 1024 * 4096
    df_plot["tokens"] = df_plot["n_iter"] * df_plot['global_batch_size'] * df_plot['seq_length']
    # print(df_plot["tokens"].max())
    # print(df_plot.benchmark.value_counts())
    
    # df_plot.dataset = df_plot.dataset.loc[df_plot.dataset.isin(["MixtureVitae", "MixtureVitae-wo_dnm"])]
    return df_plot
    
df_plot = data_load_filtering(model_size = '1.7b', n_tokens = '300B', seq_length = 4096, il_warmup_iters = 25000)
print("df_plot.benchmark.value_counts():", df_plot.benchmark.value_counts())
print("df_plot.dataset.value_counts():", df_plot.dataset.value_counts())

sizes = [1.7]
fig, axes = plt.subplots(1, len(sizes), figsize=(10, 8))
if len(sizes) == 1:
    axes = [axes]
for i, (ax, size) in enumerate(zip(axes, sizes)):
    print(df_plot.benchmark.value_counts())
    df_sub = df_plot.loc[(df_plot['size'] == size) & (df_plot.benchmark.isin(bench_sel)), :].copy()
    # print(df_sub.benchmark.value_counts())
    # print(bench_sel)
    df_iter = df_sub.pivot_table(index=["dataset", "tokens"], columns="benchmark", values="value").loc[:, bench_sel].mean(axis=1)
    # print(df_iter)
    df_iter_pivot = df_iter.reset_index().pivot_table(index="tokens", columns="dataset", values=0)
    # print(df_iter_pivot)
    df_iter_pivot = df_iter_pivot.dropna(how="any") # drop na tokens (for all datasets)
    # dataset_order = [
    #     "MixtureVitae",
    #     "MixtureVitae-wo_dnm",
    # ] 
    # df_iter_pivot.plot(
    #     ax=ax, #marker="."
    # )

    for j, col in enumerate(df_iter_pivot.columns):
        print(col)
        ax.plot(
            df_iter_pivot.index,
            df_iter_pivot[col],
            label=col,
            color=datasets[col]['color'],
            linestyle=datasets[col]['linestyle'],
            linewidth=2.8
        )
        y_position = df_iter_pivot[col].values.tolist()[-1]
        if col == "wo_tier2b":
            y_position = y_position + 0.005
        # x_position = max(df_iter_pivot.index.tolist()) + 0.02 * 1e11
        x_position = max(df_iter_pivot.index.tolist()) - 0.2 * 1e11
        ax.text(
            x_position,  # X-position (just past the end of the line at 3.0)
            y_position, # Y-position (matches the last data point)
            datasets[col]['name'],          # The text label
            color=datasets[col]['color'],
            fontsize=17,
            
            # fontweight=font_weight,
            fontweight="bold",
            ha='right',          # Horizontal alignment
            va='center'         # Vertical alignment
        )

    ax.grid(True, linestyle=':', which='both', color='#aaa', alpha=0.7)
    # ax.set_title(f"Model Size:{size} B", fontsize=20,fontweight="bold");
    ax.set_xlabel(f"Number of tokens (300 billions)", fontsize=24);
    # ax.set_xlabel("Number of iterations");
    #ax.set_ylabel(f"Average performance on {len(bench_sel)} tasks");
    ax.set_ylabel(f"Average downstream performance", fontsize=24);
    ax.set_xlim(0, 3.0 * 1e11)
    ax.set_ylim(bottom=None, top=0.60)  # Extends only upward
    # ax.set_xticklabels(["0.0","0.2","0.4","0.6","0.8","1.0","1.2","1.4","1.6","1.8","2.0","2.2","2.4","2.6","2.8","3.0"])
    tick_positions = [i * 0.5 * 1e11 for i in range(7)]  # 0, 0.2e11, 0.4e11, ... 3.0e11
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["0.0","0.5","1.0","1.5","2.0","2.5","3.0"])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    # texts = []
    # for d in datasets.values():
    #     x_position = final_x_values[d['name']] + 0.05* 1e11
    #     y_position = final_y_values[d['name']]
            
    #     texts.append(ax.text(
    #         x_position,  # X-position (just past the end of the line at 3.0)
    #         y_position, # Y-position (matches the last data point)
    #         ,          # The text label
    #         # color=label_color,
    #         fontsize=12,
    #         # fontweight=font_weight,
    #         ha='left',          # Horizontal alignment
    #         va='center'         # Vertical alignment
    #     ))
        
# rightmost_ax = axes[-1]
# pos = rightmost_ax.get_position() 

# # lines, labels = zip(*sorted(zip(lines, labels), key=lambda x: x[1]))
# leg = fig.legend(
#     # lines, [datasets[i]['name'] for i in labels],
#     loc='lower right',                # anchor point of legend box
#     bbox_to_anchor=(pos.x1+0.05, pos.y0 - 0.35 ),  # x1 = right edge, y0-0.02 = just below
#     # ncol=len(labels),
#     fontsize=20
# )


# Update legend text properties
# for text in leg.get_texts():
#     # text.set_fontsize(22)
#     text.set_color(name_color_map[text.get_text()])
#     if text.get_text() == "MixtureVitae (Ours)":
#         text.set_fontweight("bold")
#         text.set_color("black")

plt.savefig(figure_path() / "300B_mxv_decontam.pdf", bbox_inches='tight', dpi=900)

# plt.tight_layout()
# plt.show()