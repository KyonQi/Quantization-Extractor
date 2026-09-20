"""Diminishing returns when scaling MCU count, and how block+halo_on holds up.

Two-panel figure:
  - Top:    speedup curves with a SINGLE GLOBAL baseline = T[layer-off, N=2].
            All four configs are normalized to the same anchor, so block-on
            stays on top at every N — showing it always delivers the lowest
            absolute latency.
  - Bottom: parallel efficiency (per-config) = (T[cfg, N=2] / T[cfg, N]) / (N/2).
            Shows how well each individual config keeps up with ideal linear
            scaling. Closer to 100% = better scaling.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_ROOT = Path("/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay")
OUT_PATH = Path(__file__).parent / "01_diminishing_returns.png"

MCUS = [2, 3, 4, 5, 6]
CONFIGS = [
    ("block", "off"),
    ("block", "on"),
    ("layer", "off"),
    ("layer", "on"),
]

COLORS = {
    ("block", "off"): "#5dade2",
    ("block", "on"): "#1f4e79",
    ("layer", "off"): "#f1948a",
    ("layer", "on"): "#922b21",
}
MARKERS = {
    ("block", "off"): "o",
    ("block", "on"): "s",
    ("layer", "off"): "v",
    ("layer", "on"): "D",
}


def read_total_seconds(n: int, mode: str, halo: str) -> float:
    candidates = [
        LOG_ROOT / f"{n}_mcus" / f"halo_{halo}" / f"coordinator_{mode}_phase_info.log",
        LOG_ROOT / f"{n}_mcus" / f"halo_{halo}" / f"coordinator_{mode}_phase.log",
    ]
    for p in candidates:
        if p.exists():
            m = re.search(r"Inference completed in ([\d.]+) seconds", p.read_text())
            if m:
                return float(m.group(1))
    raise FileNotFoundError(f"No phase log for {n}_mcus / {mode} / halo_{halo}")


def main() -> None:
    totals = {cfg: np.array([read_total_seconds(n, *cfg) for n in MCUS])
              for cfg in CONFIGS}

    # global anchor: layer-off at N=2
    t_global = totals[("layer", "off")][0]

    speedups_global = {cfg: t_global / t for cfg, t in totals.items()}
    # Resource efficiency: same global baseline, normalized by ideal N/2 scaling.
    # 100% = "if layer-off scaled linearly from N=2, you'd be here". Above = beating it.
    efficiency = {cfg: speedups_global[cfg] / (np.array(MCUS) / MCUS[0]) * 100
                  for cfg in CONFIGS}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # ── top: speedup vs global baseline ──
    for cfg in CONFIGS:
        ax1.plot(MCUS, speedups_global[cfg], MARKERS[cfg] + "-",
                 color=COLORS[cfg], lw=2.2, ms=10,
                 label=f"{cfg[0]}-{cfg[1]}")
    ax1.set_ylabel("Speedup vs layer-off @ 2 MCUs", fontsize=11)
    ax1.set_title(
        "MBV2-0.35 — Speedup vs MCU Count (baseline = layer-off, N=2)",
        fontweight="bold", fontsize=12.5,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax1.set_xticks(MCUS)
    ax1.set_xlim(MCUS[0] - 0.3, MCUS[-1] + 0.3)

    # ── bottom: resource efficiency (same global baseline) ──
    ax2.axhline(100, color="black", lw=1.3, alpha=0.5, ls="--",
                label="Ideal  (@ 2 MCUs)")
    for cfg in CONFIGS:
        ax2.plot(MCUS, efficiency[cfg], MARKERS[cfg] + "-",
                 color=COLORS[cfg], lw=2.2, ms=10,
                 label=f"{cfg[0]}-{cfg[1]}")
    ax2.set_xlabel("Number of MCUs", fontsize=11)
    ax2.set_ylabel("Resource Efficiency  (%)", fontsize=11)
    ax2.set_title(
        "Efficiency = speedup × (2/N) × 100%   ",
        fontweight="bold", fontsize=11.5,
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=10, framealpha=0.95)
    ax2.set_xticks(MCUS)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
