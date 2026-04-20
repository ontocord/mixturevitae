"""
reasoning_scaling_v2.py — Reasoning scaling figure for MixtureVitae paper.
Two panels:
  Left:  controlled open-sci-ref retrains — reasoning scaling per dataset
  Right: published external models + MixtureVitae scaling curve for context

Data source: reasoning_scaling.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# 1. Load & prepare data

_HERE = Path(__file__).parent
_csv_path = next(p for p in [_HERE / "data" / "reasoning_scaling.csv",
                              _HERE / "reasoning_scaling.csv"] if p.exists())
df = pd.read_csv(_csv_path)

RENAME = {
    "CommonCorpus-en":  "CommonCorpus",
    "CommonCorpus-eng": "CommonCorpus",
    "Nemotron-cc-hq":   "Nemotron-CC-HQ",
    "Comma":            "Comma-0.1",
}
df["dataset_or_family"] = df["dataset_or_family"].replace(RENAME)

internal = df[df["model_type"] == "open-sci"].copy()
internal = internal.rename(columns={
    "dataset_or_family": "dataset",
    "n_params_B":        "size",
    "training_flops":    "flops",
    "avg_reasoning":     "score",
})
internal = internal.groupby(["dataset", "size", "flops"], as_index=False)["score"].mean()

external = df[df["model_type"] == "external"].copy()
external = external.rename(columns={
    "dataset_or_family": "family",
    "n_params_B":        "size",
    "training_flops":    "flops",
    "avg_reasoning":     "score",
})

def _fmt_size(b):
    if b < 0.1:
        return f"{int(round(b * 1000))}M"
    if abs(b - round(b)) < 0.05:
        return f"{int(round(b))}B"
    if b < 1:
        return f"{int(round(b * 1000))}M"
    return f"{b:.1f}B"

# 2. Dataset styling

DATASET_STYLE = {
    "MixtureVitae": dict(
        label="MixtureVitae", group="hero",
        color="#2ca02c", lw=2.4, ls="-", marker="*", ms=11, zorder=10,
    ),
    "Comma-0.1": dict(
        label="Comma-0.1", group="permissive",
        color="#1f77b4", lw=1.8, ls="-", marker="*", ms=8, zorder=8,
    ),
    "CommonCorpus": dict(
        label="CommonCorpus", group="permissive",
        color="#17becf", lw=1.6, ls="-", marker="*", ms=7, zorder=7,
    ),
    "FineWeb-Edu-1.4T": dict(
        label="FineWeb-Edu", group="focal",
        color="#e377c2", lw=1.4, ls=(0, (5, 3)), marker="*", ms=7, zorder=6,
    ),
    "DCLM": dict(
        label="DCLM", group="focal",
        color="#9467bd", lw=1.4, ls=(0, (5, 3)), marker="*", ms=7, zorder=6,
    ),
    "Nemotron-CC-HQ": dict(
        label="Nemotron-CC-HQ", group="focal",
        color="#ff7f0e", lw=1.4, ls=(0, (5, 3)), marker="*", ms=7, zorder=6,
    ),
    "HPLT-2.0": dict(
        label="HPLT-2.0", group="focal",
        color="#8c564b", lw=1.2, ls=(0, (5, 3)), marker="*", ms=6, zorder=5,
    ),
    "SlimPajama": dict(
        label="SlimPajama", group="background",
        color="#7f7f7f", lw=1.0, ls=(0, (3, 4)), marker="*", ms=6, zorder=3,
    ),
    "C4": dict(
        label="C4", group="background",
        color="#c49c94", lw=1.0, ls=(0, (3, 4)), marker="*", ms=6, zorder=3,
    ),
    "Pile": dict(
        label="Pile", group="background",
        color="#f7b6d2", lw=1.0, ls=(0, (3, 4)), marker="*", ms=6, zorder=3,
    ),
}

DRAW_ORDER = [
    "C4", "Pile", "SlimPajama",
    "HPLT-2.0", "Nemotron-CC-HQ", "FineWeb-Edu-1.4T", "DCLM",
    "CommonCorpus", "Comma-0.1",
    "MixtureVitae",
]

EXTERNAL_STYLE = {
    "Qwen2.5": dict(label="Qwen2.5", color="#555555", lw=1.8, ls=(0, (4, 2))),
    "Qwen3":   dict(label="Qwen3",   color="#8a3324", lw=1.8, ls=(0, (4, 2))),
    "SmolLM2": dict(label="SmolLM2", color="#d62728", lw=1.6, ls=(0, (2, 2))),
}

# 3. Figure layout

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.color":        "lightgray",
})

fig = plt.figure(figsize=(13, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.10,
                      left=0.065, right=0.97, top=0.91, bottom=0.20)
ax_l = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])

Y_MIN, Y_MAX = 0.02, 0.70

# 4. Left panel

label_positions = {}

for key in DRAW_ORDER:
    if key not in DATASET_STYLE:
        continue
    st = DATASET_STYLE[key]
    sub = internal[internal["dataset"] == key].sort_values("flops")
    if sub.empty:
        continue

    xs, ys = sub["flops"].tolist(), sub["score"].tolist()
    ax_l.plot(xs, ys, color=st["color"], lw=st["lw"], ls=st["ls"],
              zorder=st["zorder"])
    ax_l.scatter(xs, ys, color=st["color"], marker=st["marker"],
                 s=st["ms"]**2, zorder=st["zorder"]+1,
                 edgecolors="white", linewidths=0.5)

    if key == "DCLM":
        size_xs_top = list(xs)
        size_lbls_top = [_fmt_size(s) for s in sub["size"].tolist()]

    label_positions[st["label"]] = (xs[-1], ys[-1], st)

# Vertical guide lines at each model size
for x, lbl in zip(size_xs_top, size_lbls_top):
    ax_l.axvline(x, color="#bbbbbb", lw=0.6, ls=(0, (2, 3)), alpha=0.55, zorder=0)
    ax_l.text(x, Y_MAX - 0.005, lbl, ha="center", va="top",
              fontsize=9.5, color="#555555", zorder=15,
              bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))

X_MAX_L = internal["flops"].max()
ax_l.set_xlim(internal["flops"].min() * 0.7, X_MAX_L * 3.8)

all_labels = sorted(label_positions.items(), key=lambda kv: kv[1][1])
y_slots = np.linspace(Y_MIN + 0.02, Y_MAX - 0.06, len(all_labels))
X_LABEL = X_MAX_L * 1.55

for i, (lbl, (x_pt, y_pt, st)) in enumerate(all_labels):
    is_hero = st["group"] == "hero"
    ax_l.annotate(
        lbl,
        xy=(x_pt, y_pt), xytext=(X_LABEL, y_slots[i]),
        xycoords="data", textcoords="data",
        color=st["color"],
        fontsize=9.5 if is_hero else 8.5,
        fontweight="bold" if is_hero else "normal",
        va="center", ha="left", zorder=12,
        arrowprops=dict(arrowstyle="-", color=st["color"],
                        lw=0.6, alpha=0.45, relpos=(0, 0.5)),
    )

ax_l.set_xscale("log")
ax_l.set_ylim(Y_MIN, Y_MAX)
ax_l.set_xlabel("Training FLOPs", labelpad=4)
ax_l.set_ylabel("Avg. reasoning performance (GSM8K, IFEval, MBPP)", labelpad=4)
ax_l.set_title("Controlled experiment (open-sci-ref)", pad=6)

# 5. Right panel

right_lines = []  # (label, last_x, last_y, color, points_with_size_strs, is_hero)

for family, style in EXTERNAL_STYLE.items():
    sub = external[external["family"] == family].sort_values("flops")
    if sub.empty:
        continue
    xs = sub["flops"].tolist()
    ys = sub["score"].tolist()
    sizes = sub["size"].tolist()
    ax_r.plot(xs, ys, color=style["color"], lw=style["lw"], ls=style["ls"],
              zorder=5, marker="o", markersize=5,
              markerfacecolor=style["color"],
              markeredgewidth=0.5, markeredgecolor="white")
    pts = list(zip(xs, ys, [_fmt_size(s) for s in sizes]))
    right_lines.append((style["label"], xs[-1], ys[-1], style["color"], pts, False))

mv = internal[internal["dataset"] == "MixtureVitae"].sort_values("flops")
mv_xs, mv_ys = mv["flops"].tolist(), mv["score"].tolist()
mv_sizes = mv["size"].tolist()
mv_pts = list(zip(mv_xs, mv_ys, [_fmt_size(s) for s in mv_sizes]))
ax_r.plot(mv_xs, mv_ys, color="#2ca02c", lw=2.4, ls="-", zorder=10,
          marker="*", markersize=10,
          markerfacecolor="#2ca02c", markeredgewidth=0.5, markeredgecolor="white")
right_lines.append(("MixtureVitae", mv_xs[-1], mv_ys[-1], "#2ca02c", mv_pts, True))

all_x = [p[0] for *_, pts, _ in right_lines for p in pts]
X_R_MIN, X_R_MAX = min(all_x), max(all_x)

ax_r.set_xscale("log")
ax_r.set_xlim(X_R_MIN * 0.12, X_R_MAX * 18)
ax_r.set_ylim(Y_MIN, Y_MAX)

# Slot-based right-margin family labels with leader lines (matches left panel style)
right_lines_sorted = sorted(right_lines, key=lambda t: t[2])
y_slots = np.linspace(Y_MIN + 0.04, Y_MAX - 0.04, len(right_lines_sorted))
X_LABEL_R = X_R_MAX * 4.5

for (lbl, last_x, last_y, color, pts, is_hero), y_pos in zip(right_lines_sorted, y_slots):
    ax_r.annotate(
        lbl, xy=(last_x, last_y),
        xytext=(X_LABEL_R, y_pos),
        xycoords="data", textcoords="data",
        color=color,
        fontsize=10 if is_hero else 9,
        fontweight="bold" if is_hero else "normal",
        va="center", ha="left", zorder=12,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.55,
                        relpos=(0, 0.5)),
    )

# Size labels: every point labelled using a greedy collision-aware placer.
# Each label is placed in one of four quadrants (UL, UR, LR, LL) around its
# marker — never on the line. We try positions in order and keep the first
# that doesn't overlap any already-placed label's pixel bbox.
QUADRANTS = [
    dict(off=(-8, 11),  ha="right",  va="bottom"),  # 0: UL
    dict(off=(8, -11),  ha="left",   va="top"),     # 1: LR
    dict(off=(8, 11),   ha="left",   va="bottom"),  # 2: UR
    dict(off=(-8, -11), ha="right",  va="top"),     # 3: LL
    dict(off=(-8, 90),  ha="right",  va="bottom"),  # 4: UL, lifted (MixtureVitae 130M)
    dict(off=(0, -11),  ha="center", va="top"),     # 5: S (directly below)
    dict(off=(8, 18),   ha="left",   va="bottom"),  # 6: UR, slight lift (SmolLM2 130M)
    dict(off=(-8, 26),  ha="right",  va="bottom"),  # 7: UL, slight lift (MixtureVitae 1.7B)
]

def _label_box(ax, data_xy, off, ha, va, text):
    px_per_char = 4.5
    w = max(len(text) * px_per_char, 16)
    h = 9
    x_px, y_px = ax.transData.transform(data_xy)
    x_px += off[0]
    y_px += off[1]
    if ha == "right":
        x_px -= w
    elif ha == "center":
        x_px -= w / 2
    if va == "top":
        y_px -= h
    elif va == "center":
        y_px -= h / 2
    return (x_px, y_px, x_px + w, y_px + h)

def _rect_overlap(r1, r2):
    return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])

# Per-family forced label side.
FORCED_SIDE = {
    "Qwen2.5":     QUADRANTS[0],
    "MixtureVitae": QUADRANTS[0],
    "Qwen3":       QUADRANTS[1],
    "SmolLM2":     QUADRANTS[1],
}

# Per-point overrides (family_label, size_str) -> quadrant. Used to flip
# specific labels that would otherwise collide with neighbours.
POINT_OVERRIDES = {
    ("MixtureVitae", "1.7B"): QUADRANTS[7],   # UL-lifted, off 1.3B
    ("Qwen2.5",      "3B"):   QUADRANTS[1],   # LR, user request
    ("SmolLM2",      "1.7B"): QUADRANTS[0],   # UL, avoid Qwen3 0.6B
    ("SmolLM2",      "130M"): QUADRANTS[0],   # UL, off the line (line ascends to the right)
    ("Qwen3",        "600M"): QUADRANTS[5],   # S (directly below), into clear space
}

placed_boxes = []
for fam_idx, r in enumerate(right_lines):
    lbl, last_x, last_y, color, pts, is_hero = r
    forced_default = FORCED_SIDE.get(lbl)
    for i, (x, y, sz) in enumerate(pts):
        forced = POINT_OVERRIDES.get((lbl, sz), forced_default)
        if forced is not None:
            cfg = forced
            box = _label_box(ax_r, (x, y), cfg["off"], cfg["ha"], cfg["va"], sz)
        else:
            order = [(i + fam_idx + j) % 4 for j in range(4)]
            chosen = None
            for q in order:
                c = QUADRANTS[q]
                b = _label_box(ax_r, (x, y), c["off"], c["ha"], c["va"], sz)
                if not any(_rect_overlap(b, pb) for pb in placed_boxes):
                    chosen = (c, b)
                    break
            if chosen is None:
                c = QUADRANTS[order[0]]
                chosen = (c, _label_box(ax_r, (x, y), c["off"], c["ha"], c["va"], sz))
            cfg, box = chosen
        placed_boxes.append(box)
        ax_r.annotate(
            sz, xy=(x, y), xytext=cfg["off"],
            textcoords="offset points",
            fontsize=7, color=color,
            ha=cfg["ha"], va=cfg["va"],
            fontweight="bold" if is_hero else "normal",
            zorder=15,
            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                      ec="none", alpha=0.92),
        )

ax_r.set_xlabel("Training FLOPs", labelpad=4)
ax_r.yaxis.set_tick_params(labelleft=True)
ax_r.set_title("Published external models", pad=6)

# 6. Shared legend

legend_handles = [
    Line2D([0],[0], color="#2ca02c", lw=2.4, ls="-",
           label="MixtureVitae (ours)"),
    Line2D([0],[0], color="#1f77b4", lw=1.8, ls="-",
           label="Permissive baselines"),
    Line2D([0],[0], color="#9467bd", lw=1.4, ls=(0,(5,3)),
           label="Mixed-license baselines"),
    Line2D([0],[0], color="#555555", lw=1.8, ls=(0,(4,2)),
           label="Published external models"),
]
fig.legend(handles=legend_handles,
           loc="lower center", ncol=4, fontsize=11,
           framealpha=0.9, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, 0.02), handlelength=2.2)

# 7. Save

out_dir = Path(__file__).parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "compute_mv_reasoning_scale_v2_out.png"
out_pdf = out_dir / "compute_mv_reasoning_scale_v2_out.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
