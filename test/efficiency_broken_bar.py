"""
Parallel Efficiency — Broken-axis grouped bar chart (standalone).
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    t = sum(x["total"] for x in d)
    c = sum(x["compute"] for x in d)
    return t, c

KEYS = {"035": "MBV2-0.35", "mcunet": "MCUNet",
        "mnasnet": "MNASNet-0.5", "proxy": "ProxylessNAS"}
MCUS = [3, 4, 5, 6]

records = []
for mkey, mname in KEYS.items():
    for n in MCUS:
        try:
            tot, comp = aggregate(f"./test/{n}_mcus/coordinator_{mkey}_pw_layer.log")
            records.append(dict(model=mname, n_mcu=n, total=tot, compute=comp))
        except FileNotFoundError:
            pass
df = pd.DataFrame(records)

single_compute = {}
for mname in MODEL_NAMES:
    vals = [r.iloc[0]["compute"] * r.iloc[0]["n_mcu"]
            for n in MCUS
            for r in [df[(df.model == mname) & (df.n_mcu == n)]]
            if not r.empty]
    if vals:
        single_compute[mname] = np.mean(vals)

eff_data = {}
for mname in MODEL_NAMES:
    sub = df[df.model == mname].sort_values("n_mcu")
    if mname in single_compute and not sub.empty:
        sc = single_compute[mname]
        eff_data[mname] = (sc / sub.n_mcu.values) / sub.total.values * 100

# ── Figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.2))

n_models = len(MODEL_NAMES)
x_vals = np.arange(len(MCUS))
bar_w = 0.18

for i, mname in enumerate(MODEL_NAMES):
    if mname not in eff_data:
        continue
    offset = (i - (n_models - 1) / 2) * bar_w
    bars = ax.bar(x_vals + offset, eff_data[mname],
                  width=bar_w * 0.88, color=PAL[mname],
                  edgecolor="white", linewidth=0.3,
                  label=mname, zorder=3)

# Value annotations
for i, mname in enumerate(MODEL_NAMES):
    if mname not in eff_data:
        continue
    offset = (i - (n_models - 1) / 2) * bar_w
    for j, val in enumerate(eff_data[mname]):
        ax.text(x_vals[j] + offset, val + 0.4,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=5.5, color=PAL[mname], fontweight="bold")

ax.set_xlabel("Number of MCUs", fontsize=9.5)
ax.set_ylabel("Parallel Efficiency (%)", fontsize=9.5)
ax.set_xticks(x_vals)
ax.set_xticklabels([str(n) for n in MCUS])
ax.set_ylim(35, 72)
ax.legend(fontsize=7, loc="upper right", handlelength=1.0,
          handletextpad=0.3, borderpad=0.4, ncol=1, columnspacing=0.6)
ax.tick_params(labelsize=8.5)
sns.despine(ax=ax)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)

plt.tight_layout()
fig.savefig("./test/efficiency_broken_bar.pdf", dpi=300, bbox_inches="tight")
fig.savefig("./test/efficiency_broken_bar.png", dpi=300, bbox_inches="tight")
print("Saved to ./test/efficiency_broken_bar.{pdf,png}")
