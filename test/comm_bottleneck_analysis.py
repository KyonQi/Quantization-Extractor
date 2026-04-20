"""
Communication Bottleneck Analysis for Multi-MCU Distributed Inference
Publication-quality 3-panel figure (CVPR/NeurIPS style).

(a) Parallel efficiency degrades as MCU count increases
(b) Communication dominates total latency and ratio rises with MCU count
(c) Block fusion trades marginal compute overhead for significant comm reduction
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ── Style ────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font_scale=1.0)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.35,
    "grid.alpha": 0.25,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    "legend.edgecolor": "0.85",
    "legend.fancybox": False,
    "legend.framealpha": 0.95,
    "figure.dpi": 150,
})

# ── Palette ──────────────────────────────────────────────────────────
MODEL_NAMES = ["MBV2-0.35", "MCUNet", "MNASNet-0.5", "ProxylessNAS"]
# Muted academic palette
PAL = {
    "MBV2-0.35":    "#3778BF",  # steel blue
    "MCUNet":       "#D45E5E",  # muted red
    "MNASNet-0.5":  "#5EA86B",  # forest green
    "ProxylessNAS": "#E8943A",  # amber
}
MARKERS = {"MBV2-0.35": "o", "MCUNet": "s", "MNASNet-0.5": "D", "ProxylessNAS": "^"}

COMP_COLOR = "#5B8DB8"   # calm blue
COMM_COLOR = "#C75C5C"   # calm red


# ── Data Parsing ─────────────────────────────────────────────────────
def parse_log(filename):
    data = []
    pat = (r"Layer\s+([\d-]+)\s+\[\s*(\w+)\]\s+([\w\+]+):\s+"
           r"total=([\d.]+)ms\s+compute=([\d.]+)ms")
    with open(filename) as f:
        for line in f:
            m = re.search(pat, line)
            if m:
                data.append({"total": float(m.group(4)), "compute": float(m.group(5))})
    return data


def aggregate(filename):
    data = parse_log(filename)
    total = sum(d["total"] for d in data)
    compute = sum(d["compute"] for d in data)
    return total, compute, total - compute


KEYS = {"035": "MBV2-0.35", "mcunet": "MCUNet",
        "mnasnet": "MNASNet-0.5", "proxy": "ProxylessNAS"}
MCUS = [3, 4, 5, 6]

records = []
for mkey, mname in KEYS.items():
    for n in MCUS:
        for mode in ["layer", "block"]:
            try:
                tot, comp, comm = aggregate(
                    f"./test/{n}_mcus/coordinator_{mkey}_pw_{mode}.log")
                records.append(dict(model=mname, n_mcu=n, comm_mode=mode,
                                    total=tot, compute=comp, comm=comm,
                                    comm_pct=comm / tot * 100))
            except FileNotFoundError:
                pass
df = pd.DataFrame(records)

# Estimate single-MCU compute = sum of all compute across layers (same regardless of MCU count)
# Use the layer-mode data; pick the MCU count that has the most stable data
single_compute = {}
for mname in MODEL_NAMES:
    # Total compute is invariant to MCU count; use all available and take the max
    # (max because fewer MCUs = less overhead in scheduling, closest to true total)
    candidates = []
    for n in MCUS:
        r = df[(df.model == mname) & (df.n_mcu == n) & (df["comm_mode"] == "layer")]
        if not r.empty:
            candidates.append(r.iloc[0]["compute"] * n)
    if candidates:
        single_compute[mname] = np.mean(candidates)


# ── Figure with broken axis for panel (a) ────────────────────────────
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14.5, 3.8))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.15],
              height_ratios=[1, 3.5], hspace=0.08,
              wspace=0.30, left=0.05, right=0.97, bottom=0.16, top=0.88)

ax_top = fig.add_subplot(gs[0, 0])   # top part: ideal bar (100%)
ax_bot = fig.add_subplot(gs[1, 0])   # bottom part: actual bars (35-65%)
axes = [None, fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]

# =====================================================================
# (a) Parallel Efficiency — Broken-axis grouped bars
# =====================================================================
# Compute efficiency data
eff_data = {}
for mname in MODEL_NAMES:
    sub = df[(df.model == mname) & (df["comm_mode"] == "layer")].sort_values("n_mcu")
    if mname not in single_compute or sub.empty:
        continue
    sc = single_compute[mname]
    eff_data[mname] = (sc / sub.n_mcu.values) / sub.total.values * 100

n_models = len(MODEL_NAMES)
n_mcus = len(MCUS)
x_vals = np.arange(n_mcus)
bar_w = 0.18

for ax_part in [ax_top, ax_bot]:
    for i, mname in enumerate(MODEL_NAMES):
        if mname not in eff_data:
            continue
        offset = (i - (n_models - 1) / 2) * bar_w
        bars = ax_part.bar(x_vals + offset, eff_data[mname],
                           width=bar_w * 0.88, color=PAL[mname],
                           edgecolor="white", linewidth=0.3,
                           label=mname if ax_part is ax_bot else "", zorder=3)

    # Ideal reference bar (100%)
    ax_part.axhline(100, color="0.5", ls="--", lw=0.7, zorder=1)

# ── Broken axis ranges ──
ax_top.set_ylim(95, 105)
ax_bot.set_ylim(35, 68)

# Hide spines between the two
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(bottom=False, labelbottom=False)
ax_top.set_xticks([])

# Diagonal break marks
d = 0.012
kwargs = dict(transform=ax_top.transAxes, color="0.4", clip_on=False, lw=0.7)
ax_top.plot((-d, +d), (-d * 4, +d * 4), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d * 4, +d * 4), **kwargs)
kwargs["transform"] = ax_bot.transAxes
ax_bot.plot((-d, +d), (1 - d * 4, 1 + d * 4), **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d * 4, 1 + d * 4), **kwargs)

# "Ideal" label
ax_top.text(n_mcus - 0.5 + 0.15, 100, "Ideal", fontsize=7, color="0.45",
            va="center", ha="left")

# Value annotations on bottom bars
for i, mname in enumerate(MODEL_NAMES):
    if mname not in eff_data:
        continue
    offset = (i - (n_models - 1) / 2) * bar_w
    for j, val in enumerate(eff_data[mname]):
        ax_bot.text(x_vals[j] + offset, val + 0.6,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=5, color=PAL[mname], fontweight="bold")

ax_bot.set_xlabel("Number of MCUs", fontsize=9)
ax_bot.set_ylabel("Parallel Efficiency (%)", fontsize=9)
ax_bot.yaxis.set_label_coords(-0.12, 0.65)  # center label across both axes
ax_top.set_title("(a) Parallel Efficiency", fontsize=10, fontweight="bold", pad=6)
ax_bot.set_xticks(x_vals)
ax_bot.set_xticklabels([str(n) for n in MCUS])
ax_bot.legend(fontsize=6.5, loc="upper right", handlelength=1.0,
              handletextpad=0.3, borderpad=0.4, ncol=2, columnspacing=0.6)
ax_bot.tick_params(labelsize=8)
ax_top.tick_params(labelsize=8)
sns.despine(ax=ax_bot, top=True)
sns.despine(ax=ax_top, bottom=True)
ax_bot.yaxis.grid(True, linewidth=0.3, alpha=0.25)
ax_top.yaxis.grid(True, linewidth=0.3, alpha=0.25)

# =====================================================================
# (b) Communication Ratio (Grouped Bars)
# =====================================================================
ax = axes[1]

layer_df = df[df.comm_mode == "layer"].copy()
n_models = len(MODEL_NAMES)
bar_w = 0.17
x_vals = np.arange(len(MCUS))

for i, mname in enumerate(MODEL_NAMES):
    sub = layer_df[layer_df.model == mname].sort_values("n_mcu")
    if sub.empty:
        continue
    offset = (i - (n_models - 1) / 2) * bar_w
    bars = ax.bar(x_vals + offset, sub.comm_pct.values,
                  width=bar_w * 0.88, color=PAL[mname],
                  edgecolor="white", linewidth=0.3,
                  label=mname, zorder=3)
    # Value labels on top
    for bar, val in zip(bars, sub.comm_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.0f}", ha="center", va="bottom", fontsize=5.5,
                color=PAL[mname], fontweight="bold")

ax.axhline(50, color=COMM_COLOR, ls=":", lw=0.7, alpha=0.45)
ax.text(3.48, 51.2, "50%", fontsize=6.5, color=COMM_COLOR, alpha=0.6)

ax.set_xlabel("Number of MCUs", fontsize=9)
ax.set_ylabel("Comm. / Total Latency (%)", fontsize=9)
ax.set_title("(b) Communication Ratio", fontsize=10, fontweight="bold", pad=6)
ax.set_xticks(x_vals)
ax.set_xticklabels([str(n) for n in MCUS])
ax.set_ylim(0, 65)
ax.legend(fontsize=6.5, loc="upper left", ncol=2,
          handlelength=1.0, handletextpad=0.3, columnspacing=0.6, borderpad=0.4)
ax.tick_params(labelsize=8)
sns.despine(ax=ax)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.3)

# =====================================================================
# (c) Block Fusion: Stacked Bars — Layer vs Block
# =====================================================================
ax = axes[2]
target_n = 5

x_pos = np.arange(len(MODEL_NAMES))
bw = 0.30  # bar width
gap = 0.04

for i, mname in enumerate(MODEL_NAMES):
    for j, (mode, label_m) in enumerate([("layer", "Layer"), ("block", "Block")]):
        row = df[(df.model == mname) & (df.n_mcu == target_n) & (df["comm_mode"] == mode)]
        if row.empty:
            continue
        comp_v = row.iloc[0]["compute"]
        comm_v = row.iloc[0]["comm"]
        xc = x_pos[i] + (j - 0.5) * (bw + gap)

        hatch = "" if mode == "layer" else "///"
        alpha = 1.0 if mode == "layer" else 0.75

        ax.bar(xc, comp_v, width=bw, color=COMP_COLOR, alpha=alpha,
               edgecolor="white", linewidth=0.3, hatch=hatch, zorder=3)
        ax.bar(xc, comm_v, width=bw, bottom=comp_v, color=COMM_COLOR,
               alpha=alpha, edgecolor="white", linewidth=0.3, hatch=hatch, zorder=3)

    # Annotate total reduction
    lr = df[(df.model == mname) & (df.n_mcu == target_n) & (df["comm_mode"] == "layer")]
    br = df[(df.model == mname) & (df.n_mcu == target_n) & (df["comm_mode"] == "block")]
    if lr.empty or br.empty:
        continue
    l_total = lr.iloc[0]["total"]
    b_total = br.iloc[0]["total"]
    l_comm = lr.iloc[0]["comm"]
    b_comm = br.iloc[0]["comm"]
    total_red = (l_total - b_total) / l_total * 100
    comm_red = (l_comm - b_comm) / l_comm * 100

    # Arrow from layer-bar top down to block-bar top
    layer_x = x_pos[i] + (-0.5) * (bw + gap)
    block_x = x_pos[i] + (0.5) * (bw + gap)
    mid_x = x_pos[i]

    ax.annotate(
        "",
        xy=(block_x, b_total), xytext=(layer_x, l_total),
        arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                        color="0.3", lw=1.0, connectionstyle="arc3,rad=-0.15"),
        zorder=5,
    )
    ax.text(mid_x, l_total + l_total * 0.03,
            f"$\\downarrow${total_red:.0f}%",
            fontsize=7, fontweight="bold", color="#333",
            ha="center", va="bottom", zorder=5)

# Legend
legend_elems = [
    Patch(fc=COMP_COLOR, ec="white", label="Compute"),
    Patch(fc=COMM_COLOR, ec="white", label="Communication"),
    Patch(fc="0.7", ec="0.4", label="Layer mode"),
    Patch(fc="0.7", ec="0.4", hatch="///", label="Block mode"),
]
ax.legend(handles=legend_elems, fontsize=6.5, loc="upper right",
          handlelength=1.2, handletextpad=0.3, ncol=2,
          columnspacing=0.6, borderpad=0.4)

ax.set_xlabel("Model", fontsize=9)
ax.set_ylabel("Latency (ms)", fontsize=9)
ax.set_title(f"(c) Block Fusion ({target_n} MCUs)", fontsize=10,
             fontweight="bold", pad=6)
ax.set_xticks(x_pos)
ax.set_xticklabels(["MBV2\n0.35", "MCU\nNet", "MNAS\nNet-0.5", "Proxyless\nNAS"],
                    fontsize=7)
ax.tick_params(labelsize=8)
sns.despine(ax=ax)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.3)

# ── Save ─────────────────────────────────────────────────────────────
fig.savefig("./test/comm_bottleneck_analysis.pdf", dpi=300, bbox_inches="tight")
fig.savefig("./test/comm_bottleneck_analysis.png", dpi=300, bbox_inches="tight")
print("Saved to ./test/comm_bottleneck_analysis.{pdf,png}")
