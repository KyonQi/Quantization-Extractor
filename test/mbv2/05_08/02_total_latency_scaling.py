"""Total inference latency vs MCU count (the "hero" plot).

Single figure with 4 lines — one per (mode, halo) configuration.
Clean styling: no per-point text annotations, no callouts; let the legend
and grid lines speak.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_ROOT = Path("/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay")
OUT_PATH = Path(__file__).parent / "02_total_latency_scaling.png"

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
    times_ms = {cfg: np.array([read_total_seconds(n, *cfg) * 1000 for n in MCUS])
                for cfg in CONFIGS}

    fig, ax = plt.subplots(figsize=(11.5, 7))

    for cfg in CONFIGS:
        ax.plot(MCUS, times_ms[cfg], MARKERS[cfg] + "-",
                color=COLORS[cfg], lw=2.2, ms=10,
                label=f"{cfg[0]} - halo {cfg[1]}")

    ax.set_xlabel("Number of MCUs", fontsize=11)
    ax.set_ylabel("Total inference time  (ms)", fontsize=11)
    ax.set_title(
        "MBV2-0.35 — Total Inference Latency vs MCU Count",
        fontweight="bold", fontsize=14,
    )
    ax.set_xticks(MCUS)
    ax.set_xlim(MCUS[0] - 0.3, MCUS[-1] + 0.3)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
