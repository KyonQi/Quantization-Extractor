"""Compare per-layer latency before/after enabling TCP_QUICKACK on coord.

Both runs use halo_on. Highlights how Linux Delayed-ACK was inflating per-layer
latency (most visibly init_conv, ~40ms) and how QUICKACK eliminates it.

Outputs (PNG, into --out-dir):
    halo_on_nodelay_total.png       — total time per layer, before vs after
    halo_on_nodelay_breakdown.png   — net_oh per layer, before vs after

Usage:
    python -m test.mbv2.mbv2_035_nodelay_comparison [--up-to blk4]
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BEFORE = Path(
    "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_on/bugs_in_init/coordinator_block_info.log"
)
DEFAULT_AFTER = Path(
    "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_on/coordinator_block_info.log"
)
DEFAULT_OUT_DIR = Path(
    "/home/kyonqi/Project/RustProjects/Python_Sim_Infer/test/mbv2/halo"
)


# Flexible regex: send(avg/max) is optional — present in newer logs only.
LAYER_RE = re.compile(
    r"Layer\s+(?P<idx>\S+)\s+\[\s*(?P<type>\S+)\s*\]\s+(?P<name>\S+):\s+"
    r"total=\s*(?P<total>[\d.]+)ms\s+"
    r"(?:send\(avg/max\)=\s*(?P<send_avg>[\d.]+)/\s*(?P<send_max>[\d.]+)ms\s+)?"
    r"cmpt\(avg/max\)=\s*(?P<cmpt_avg>[\d.]+)/\s*(?P<cmpt_max>[\d.]+)ms\s+"
    r"wait\(avg/max\)=\s*(?P<wait_avg>[\d.]+)/\s*(?P<wait_max>[\d.]+)ms\s+"
    r"net_oh\(avg/max\)=\s*(?P<oh_avg>[\d.]+)/\s*(?P<oh_max>[\d.]+)ms\s+"
)


@dataclass
class LayerRow:
    idx: str
    type_: str
    name: str
    total_ms: float
    cmpt_max_ms: float
    wait_max_ms: float
    oh_max_ms: float


def parse_log(path: Path) -> list[LayerRow]:
    rows: list[LayerRow] = []
    for line in path.read_text().splitlines():
        m = LAYER_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            LayerRow(
                idx=d["idx"],
                type_=d["type"],
                name=d["name"],
                total_ms=float(d["total"]),
                cmpt_max_ms=float(d["cmpt_max"]),
                wait_max_ms=float(d["wait_max"]),
                oh_max_ms=float(d["oh_max"]),
            )
        )
    if not rows:
        raise ValueError(f"No layer lines parsed from {path} — check LAYER_RE")
    return rows


def short_label(r: LayerRow) -> str:
    if r.type_ == "BLOCK":
        base = r.name.split("_")[0]
        return f"{base}\n({r.idx})"
    return f"{r.name}\n({r.idx})"


# ─────────────────────────── plotting ────────────────────────────────


def plot_total_time(before: list[LayerRow], after: list[LayerRow], out_path: Path) -> None:
    n = len(before)
    x = np.arange(n)
    w = 0.4
    bef = np.array([r.total_ms for r in before])
    aft = np.array([r.total_ms for r in after])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w / 2, bef, w, label="before fix (Delayed-ACK)", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, aft, w, label="after fix (TCP_QUICKACK)", color="#2ca02c", alpha=0.85)

    y_max = max(bef.max(), aft.max())
    pad = y_max * 0.02
    for i in range(n):
        delta = bef[i] - aft[i]
        if abs(delta) >= 1.0:
            pct = delta / bef[i] * 100 if bef[i] > 0 else 0
            ax.annotate(
                f"-{delta:.1f}ms\n({pct:+.0f}%)",
                xy=(i, max(bef[i], aft[i]) + pad),
                ha="center", va="bottom",
                fontsize=8, color="darkgreen", fontweight="bold",
            )
    ax.set_ylim(top=y_max * 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in before], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Layer total time (ms)")
    ax.set_title(
        "Effect of TCP_QUICKACK on per-layer latency (halo_on, MBV2-0.35, 4 MCUs)",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_breakdown(before: list[LayerRow], after: list[LayerRow], out_path: Path) -> None:
    """Stack compute + net_oh; show net_oh % drop annotation per layer."""
    n = len(before)
    x = np.arange(n)
    w = 0.4
    bef_cmpt = np.array([r.cmpt_max_ms for r in before])
    aft_cmpt = np.array([r.cmpt_max_ms for r in after])
    # Use total - compute as the "non-compute" overhead (captures coord side too)
    bef_nc = np.array([max(0.0, r.total_ms - r.cmpt_max_ms) for r in before])
    aft_nc = np.array([max(0.0, r.total_ms - r.cmpt_max_ms) for r in after])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w / 2, bef_cmpt, w, color="#1f77b4", label="compute (max)")
    ax.bar(x - w / 2, bef_nc, w, bottom=bef_cmpt, color="#d62728", label="non-compute BEFORE")
    ax.bar(x + w / 2, aft_cmpt, w, color="#1f77b4")
    ax.bar(x + w / 2, aft_nc, w, bottom=aft_cmpt, color="#2ca02c", label="non-compute AFTER")

    bef_total = bef_cmpt + bef_nc
    aft_total = aft_cmpt + aft_nc
    y_max = max(bef_total.max(), aft_total.max())
    pad = y_max * 0.02
    for i in range(n):
        if bef_nc[i] > 1e-3:
            pct = (bef_nc[i] - aft_nc[i]) / bef_nc[i] * 100
            color = "darkgreen" if pct > 0 else "red"
            top = max(bef_total[i], aft_total[i])
            ax.text(
                x[i], top + pad, f"net ↓{pct:.0f}%",
                ha="center", va="bottom",
                fontsize=8.5, color=color, fontweight="bold",
            )
    ax.set_ylim(top=y_max * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in before], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        "Compute vs non-compute — left bar = BEFORE fix, right = AFTER fix",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_summary(before: list[LayerRow], after: list[LayerRow]) -> None:
    tot_b = sum(r.total_ms for r in before)
    tot_a = sum(r.total_ms for r in after)
    saved = tot_b - tot_a
    print(f"Total time: BEFORE={tot_b:.2f}ms  AFTER={tot_a:.2f}ms  "
          f"saved={saved:.2f}ms ({100*saved/tot_b:.1f}%)\n")
    print(f"{'layer':>16s}  {'BEFORE':>8s}  {'AFTER':>8s}  {'Δms':>9s}")
    for b, a in zip(before, after):
        delta = b.total_ms - a.total_ms
        print(f"{a.name[:16]:>16s}  {b.total_ms:>8.2f}  {a.total_ms:>8.2f}  {delta:>+9.2f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    p.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--up-to", type=str, default="blk4",
                   help="Truncate at and including this block name")
    args = p.parse_args()

    before = parse_log(args.before)
    after = parse_log(args.after)
    if [r.idx for r in before] != [r.idx for r in after]:
        raise ValueError("Layer indices don't line up between BEFORE and AFTER logs")

    if args.up_to:
        cutoff = None
        for i, r in enumerate(before):
            if args.up_to in r.name:
                cutoff = i + 1
        if cutoff is None:
            raise ValueError(f"--up-to '{args.up_to}' did not match any layer name")
        before = before[:cutoff]
        after = after[:cutoff]
        print(f"Truncated to first {cutoff} entries (up to and including '{args.up_to}')\n")

    print_summary(before, after)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_total_time(before, after, args.out_dir / "halo_on_nodelay_total.png")
    plot_breakdown(before, after, args.out_dir / "halo_on_nodelay_breakdown.png")


if __name__ == "__main__":
    main()
