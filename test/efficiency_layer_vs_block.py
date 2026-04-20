"""
Parallel Efficiency — Layer vs Block mode comparison.
Two side-by-side panels with shared y-axis for direct visual comparison.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

PAL = {
    "MBV2-0.35":    "#3778BF",
    "MCUNet":       "#D45E5E",
    "MNASNet-0.5":  "#5EA86B",
    "ProxylessNAS": "#E8943A",
}
MODEL_NAMES = list(PAL.keys())


# ── Parse ────────────────────────────────────────────────────────────
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


def aggregate(fn):
    d = parse_log(fn)
    return sum(x["total"] for x in d), sum(x["compute"] for x in d)


KEYS = {"035": "MBV2-0.35", "mcunet": "MCUNet",
        "mnasnet": "MNASNet-0.5", "proxy": "ProxylessNAS"}
MCUS = [3, 4, 5, 6]

records = []
for mkey, mname in KEYS.items():
    for n in MCUS:
        for mode in ["layer", "block"]:
            try:
                tot, comp = aggregate(
                    f"./test/{n}_mcus/coordinator_{mkey}_pw_{mode}.log")
                records.append(dict(model=mname, n_mcu=n, comm_mode=mode,
                                    total=tot, compute=comp))
            except FileNotFoundError:
                pass
df = pd.DataFrame(records)

# Single-MCU compute estimate from layer-mode (compute is invariant to mode/MCU)
single_compute = {}
for mname in MODEL_NAMES:
    vals = []
    for n in MCUS:
        r = df[(df.model == mname) & (df.n_mcu == n) & (df["comm_mode"] == "layer")]
        if not r.empty:
            vals.append(r.iloc[0]["compute"] * n)
    if vals:
        single_compute[mname] = np.mean(vals)

# Compute efficiency per (model, mcu, mode)
def get_eff(mname, mode):
    sub = df[(df.model == mname) & (df["comm_mode"] == mode)].sort_values("n_mcu")
    if sub.empty or mname not in single_compute:
        return None
    sc = single_compute[mname]
    return (sc / sub.n_mcu.values) / sub.total.values * 100


# ── Figure: side-by-side bar panels ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
plt.subplots_adjust(wspace=0.06)

n_models = len(MODEL_NAMES)
x_vals = np.arange(len(MCUS))
bar_w = 0.18

mode_titles = {"layer": "(a) Layer Mode", "block": "(b) Block Mode"}

for ax, mode in zip(axes, ["layer", "block"]):
    for i, mname in enumerate(MODEL_NAMES):
        eff = get_eff(mname, mode)
        if eff is None:
            continue
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(x_vals + offset, eff,
               width=bar_w * 0.88, color=PAL[mname],
               edgecolor="white", linewidth=0.3,
               label=mname, zorder=3)
        # Value labels
        for j, val in enumerate(eff):
            ax.text(x_vals[j] + offset, val + 0.5,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=5.5, color=PAL[mname], fontweight="bold")

    ax.set_xlabel("Number of MCUs", fontsize=9.5)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(n) for n in MCUS])
    ax.set_title(mode_titles[mode], fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8.5)
    sns.despine(ax=ax)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)

# Shared y range — must accommodate block mode (which is higher)
axes[0].set_ylim(35, 92)
axes[0].set_ylabel("Parallel Efficiency (%)", fontsize=9.5)

# Legend on right panel (block mode), upper right
axes[1].legend(fontsize=7, loc="upper right", handlelength=1.0,
               handletextpad=0.3, borderpad=0.4, ncol=1, columnspacing=0.6)

plt.tight_layout()
fig.savefig("./test/efficiency_layer_vs_block.pdf", dpi=300, bbox_inches="tight")
fig.savefig("./test/efficiency_layer_vs_block.png", dpi=300, bbox_inches="tight")
print("Saved to ./test/efficiency_layer_vs_block.{pdf,png}")


# ── Alt version: dumbbell chart (kept for reference) ─────────────────
# fig, ax = plt.subplots(figsize=(7.5, 3.6))
#
# n_models = len(MODEL_NAMES)
# x_base = np.arange(len(MCUS))
# group_w = 0.7
# slot_w = group_w / n_models
#
# for i, mname in enumerate(MODEL_NAMES):
#     eff_l = get_eff(mname, "layer")
#     eff_b = get_eff(mname, "block")
#     if eff_l is None or eff_b is None:
#         continue
#     x_offsets = (i - (n_models - 1) / 2) * slot_w
#     xs = x_base + x_offsets
#     for x, yl, yb in zip(xs, eff_l, eff_b):
#         ax.plot([x, x], [yl, yb], color=PAL[mname], lw=1.6,
#                 alpha=0.55, zorder=2, solid_capstyle="round")
#         delta = yb - yl
#         ax.text(x, yb + 0.6, f"+{delta:.0f}",
#                 ha="center", va="bottom", fontsize=5.5,
#                 color=PAL[mname], fontweight="bold")
#     ax.scatter(xs, eff_l, s=42, marker="o",
#                facecolors="white", edgecolors=PAL[mname],
#                linewidths=1.4, zorder=4)
#     ax.scatter(xs, eff_b, s=42, marker="o",
#                color=PAL[mname], edgecolors="white",
#                linewidths=0.6, zorder=4, label=mname)
#
# ax.set_xticks(x_base)
# ax.set_xticklabels([str(n) for n in MCUS])
# ax.set_xlabel("Number of MCUs", fontsize=9.5)
# ax.set_ylabel("Parallel Efficiency (%)", fontsize=9.5)
# ax.set_xlim(-0.5, len(MCUS) - 0.5)
# ax.set_ylim(38, 85)
# ax.tick_params(labelsize=8.5)
# sns.despine(ax=ax)
# ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)
#
# from matplotlib.lines import Line2D
# model_handles = [
#     Line2D([], [], marker="o", color=PAL[m], markerfacecolor=PAL[m],
#            markeredgecolor="white", markersize=6, lw=0, label=m)
#     for m in MODEL_NAMES
# ]
# mode_handles = [
#     Line2D([], [], marker="o", color="0.4", markerfacecolor="white",
#            markeredgecolor="0.4", markersize=6, lw=0, label="Layer"),
#     Line2D([], [], marker="o", color="0.4", markerfacecolor="0.4",
#            markeredgecolor="white", markersize=6, lw=0, label="Block"),
# ]
# leg1 = ax.legend(handles=model_handles, loc="upper right",
#                  fontsize=7, handletextpad=0.3, borderpad=0.4,
#                  ncol=1, framealpha=0.95, edgecolor="0.85")
# ax.add_artist(leg1)
# ax.legend(handles=mode_handles, loc="upper right",
#           bbox_to_anchor=(0.82, 1.0),
#           fontsize=7, handletextpad=0.3, borderpad=0.4,
#           ncol=1, framealpha=0.95, edgecolor="0.85")
