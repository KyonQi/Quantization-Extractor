"""Per-group latency breakdown — halo OFF vs ON, side by side per group.

Two panels (top = block mode, bottom = layer mode aggregated to block-equivalent
groups). Each panel shows side-by-side stacked bars per group:

   left bar  = halo OFF   (compute base = blue, net_oh in RED with 3 hatches)
   right bar = halo ON    (compute base = blue, net_oh in GREEN with 3 hatches)

Stack order from bottom to top:
   coord_overhead  (gray)
   compute         (blue, no in-segment label)
   send_dispatch   (red/green, '...' hatch)
   comm + worker_io(red/green, solid)
   post_header_recv(red/green, '///' hatch)

Layout follows mbv2_035_halo_comparison.py style.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

LOG_ROOT = Path("/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay")
OUT_PATH = Path(__file__).parent / "03_per_layer_phase_breakdown.png"

DEFAULT_MCUS = 4

GROUP_LABELS = ["init_conv", "blk0", "blk1", "blk2", "blk3", "blk4"]
LAYER_GROUPS = [
    (0, 0),
    (1, 2),
    (3, 5),
    (6, 8),
    (9, 11),
    (12, 14),
]

HEADER_RE = re.compile(
    r"Layer\s+(?P<idx>\S+)\s+\[\s*(?P<type>\S+)\s*\]\s+(?P<name>\S+):\s+"
    r"total=\s*(?P<total>[\d.]+)ms\s+"
    r"send\(avg/max\)=\s*[\d.]+/\s*[\d.]+ms\s+"
    r"cmpt\(avg/max\)=\s*(?P<cmpt_avg>[\d.]+)/\s*[\d.]+ms\s+"
    r"wait\(avg/max\)=\s*(?P<wait_avg>[\d.]+)/\s*[\d.]+ms"
)
PHASE_RE = re.compile(
    r"engine_pre=\s*(?P<engine_pre>[\d.]+)\s+"
    r"build=\s*(?P<build>[\d.]+)\s+"
    r"send_phase=\s*(?P<send_phase>[\d.]+)\s+"
    r"recv_phase=\s*(?P<recv_phase>[\d.]+)\s+"
    r"engine_post=\s*(?P<engine_post>[\d.]+)"
)


@dataclass
class Row:
    idx: str
    type_: str
    name: str
    total_ms: float
    cmpt_avg: float
    wait_avg: float
    engine_pre: float = 0.0
    build: float = 0.0
    send_phase: float = 0.0
    recv_phase: float = 0.0
    engine_post: float = 0.0


def parse_log(path: Path) -> list[Row]:
    rows: list[Row] = []
    last: Row | None = None
    for line in path.read_text().splitlines():
        h = HEADER_RE.search(line)
        if h:
            last = Row(
                idx=h["idx"], type_=h["type"], name=h["name"],
                total_ms=float(h["total"]),
                cmpt_avg=float(h["cmpt_avg"]),
                wait_avg=float(h["wait_avg"]),
            )
            rows.append(last)
            continue
        if last is None:
            continue
        p = PHASE_RE.search(line)
        if p:
            last.engine_pre = float(p["engine_pre"])
            last.build = float(p["build"])
            last.send_phase = float(p["send_phase"])
            last.recv_phase = float(p["recv_phase"])
            last.engine_post = float(p["engine_post"])
    if not rows:
        raise ValueError(f"No layer rows parsed from {path}")
    return rows


def find_log(n: int, mode: str, halo: str) -> Path:
    candidates = [
        LOG_ROOT / f"{n}_mcus" / f"halo_{halo}" / f"coordinator_{mode}_phase_info.log",
        LOG_ROOT / f"{n}_mcus" / f"halo_{halo}" / f"coordinator_{mode}_phase.log",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"{n}_mcus / {mode} / halo_{halo}")


def _row_to_segments(r: Row, label: str) -> dict:
    return {
        "label": label,
        "coord": r.engine_pre + r.build + r.engine_post,
        "cmpt": r.cmpt_avg,
        "send": r.send_phase,
        "cio": max(0.0, r.wait_avg - r.cmpt_avg),
        "pst": max(0.0, r.recv_phase - r.wait_avg),
        "total": r.total_ms,
    }


def _aggregate_rows(rows: list[Row], label: str) -> dict:
    return {
        "label": label,
        "coord": sum(r.engine_pre + r.build + r.engine_post for r in rows),
        "cmpt": sum(r.cmpt_avg for r in rows),
        "send": sum(r.send_phase for r in rows),
        "cio": sum(max(0.0, r.wait_avg - r.cmpt_avg) for r in rows),
        "pst": sum(max(0.0, r.recv_phase - r.wait_avg) for r in rows),
        "total": sum(r.total_ms for r in rows),
    }


def aggregate_into_groups(rows: list[Row], mode: str) -> list[dict]:
    out: list[dict] = []
    if mode == "block":
        for label in GROUP_LABELS:
            match: Row | None = None
            for r in rows:
                if label == "init_conv" and r.name == "init_conv":
                    match = r
                    break
                if label != "init_conv" and (
                    r.name.startswith(label + "_") or r.name.startswith(label + "+")
                ):
                    match = r
                    break
            if match is None:
                raise ValueError(f"block mode: missing entry for group '{label}'")
            out.append(_row_to_segments(match, label))
    else:
        for label, (s, e) in zip(GROUP_LABELS, LAYER_GROUPS):
            grp: list[Row] = []
            for r in rows:
                try:
                    idx = int(r.idx)
                except (ValueError, TypeError):
                    continue
                if s <= idx <= e:
                    grp.append(r)
            if not grp:
                raise ValueError(f"layer mode: no rows for group '{label}' ({s}-{e})")
            out.append(_aggregate_rows(grp, label))
    return out


# ─────────────────────── plotting ────────────────────────

COLOR_COORD = "#95a5a6"
COLOR_CMPT = "#1f77b4"
COLOR_OFF = "#d62728"
COLOR_ON = "#2ca02c"
HATCH_SEND = "..."
HATCH_CIO = ""
HATCH_POST = "///"


def draw_panel(ax, off_groups: list[dict], on_groups: list[dict], title: str,
               y_max: float):
    n = len(off_groups)
    x = np.arange(n)
    w = 0.4

    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 1.0

    # Annotate only segments that are both ≥ 1.5 ms AND ≥ 3% of the tallest bar.
    # Skip on tiny segments to avoid clutter.
    min_label_ms = max(1.5, y_max * 0.03)

    def _draw(pos, groups, oh_color):
        coord = np.array([g["coord"] for g in groups])
        cmpt = np.array([g["cmpt"] for g in groups])
        send = np.array([g["send"] for g in groups])
        cio = np.array([g["cio"] for g in groups])
        pst = np.array([g["pst"] for g in groups])
        total = np.array([g["total"] for g in groups])

        ax.bar(pos, coord, w, color=COLOR_COORD,
               edgecolor="white", linewidth=0.5, zorder=2)
        bot = coord.copy()
        ax.bar(pos, cmpt, w, bottom=bot, color=COLOR_CMPT,
               edgecolor="white", linewidth=0.5, zorder=2)
        bot += cmpt
        ax.bar(pos, send, w, bottom=bot, color=oh_color,
               hatch=HATCH_SEND, edgecolor="white", linewidth=0.5, zorder=2)
        bot += send
        ax.bar(pos, cio, w, bottom=bot, color=oh_color,
               hatch=HATCH_CIO, edgecolor="white", linewidth=0.5, zorder=2)
        bot += cio
        ax.bar(pos, pst, w, bottom=bot, color=oh_color,
               hatch=HATCH_POST, edgecolor="white", linewidth=0.5, zorder=2)

        # in-segment numbers (skip compute, threshold to keep readable)
        base = np.zeros(n)
        for arr in (coord,):
            for i, v in enumerate(arr):
                if v >= min_label_ms:
                    ax.text(pos[i], base[i] + v / 2, f"{v:.1f}",
                            ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color="white", zorder=3)
            base = base + arr
        base = base + cmpt          # skip compute label
        for arr in (send, cio, pst):
            for i, v in enumerate(arr):
                if v >= min_label_ms:
                    ax.text(pos[i], base[i] + v / 2, f"{v:.1f}",
                            ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color="white", zorder=3)
            base = base + arr

        return total

    off_total = _draw(x - w / 2, off_groups, COLOR_OFF)
    on_total = _draw(x + w / 2, on_groups, COLOR_ON)

    # net ↓N% per group above the taller of the two bars
    pad = y_max * 0.018
    for i in range(n):
        off_oh = off_groups[i]["send"] + off_groups[i]["cio"] + off_groups[i]["pst"]
        on_oh = on_groups[i]["send"] + on_groups[i]["cio"] + on_groups[i]["pst"]
        if off_oh > 1e-3:
            pct = (off_oh - on_oh) / off_oh * 100
            color = "darkgreen" if pct > 0 else "red"
            top = max(off_total[i], on_total[i])
            ax.text(x[i], top + pad, f"net ↓{pct:.0f}%",
                    ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_ylim(0, y_max * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in off_groups], fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_for_mcus(n: int) -> None:
    panels = {}
    for mode in ("block", "layer"):
        for halo in ("off", "on"):
            rows = parse_log(find_log(n, mode, halo))
            panels[(mode, halo)] = aggregate_into_groups(rows, mode)

    y_max = max(g["total"] for groups in panels.values() for g in groups)

    fig, (ax_layer, ax_block) = plt.subplots(2, 1, figsize=(13, 10), sharey=True)

    draw_panel(ax_layer,
               panels[("layer", "off")], panels[("layer", "on")],
               f"layer mode  ·  halo OFF (left bars)  vs  halo ON (right bars)  ·  {n} MCUs  "
               f"(aggregated to block-equivalent groups)",
               y_max)
    draw_panel(ax_block,
               panels[("block", "off")], panels[("block", "on")],
               f"block mode  ·  halo OFF (left bars)  vs  halo ON (right bars)  ·  {n} MCUs",
               y_max)

    ax_block.set_ylabel("Time (ms)", fontsize=11)
    ax_layer.set_ylabel("Time (ms)", fontsize=11)

    # shared legend at top: 6 entries (color + hatch meaning)
    handles = [
        Patch(facecolor=COLOR_COORD, label="coord_overhead"),
        Patch(facecolor=COLOR_CMPT, label="compute"),
        Patch(facecolor=COLOR_OFF, label="net_oh  (halo OFF)"),
        Patch(facecolor=COLOR_ON, label="net_oh  (halo ON)"),
        Patch(facecolor="#7f7f7f", hatch=HATCH_SEND, edgecolor="white",
              label="↳ send_dispatch"),
        Patch(facecolor="#7f7f7f", hatch=HATCH_CIO, edgecolor="white",
              label="↳ comm + worker_io"),
        Patch(facecolor="#7f7f7f", hatch=HATCH_POST, edgecolor="white",
              label="↳ post_header_recv"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(
        f"MBV2-0.35 — Latency Decomposition  (init_conv -> blk4, {n} MCUs)",
        fontsize=14, fontweight="bold", y=1.04,
    )

    out_path = OUT_PATH.with_name(f"03_per_layer_phase_breakdown_{n}mcus.png")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcus", type=int, default=None,
                        help="Single MCU count to plot. Omit to render all of 2,3,4,5,6.")
    args = parser.parse_args()

    target_mcus = [args.mcus] if args.mcus else [2, 3, 4, 5, 6]
    for n in target_mcus:
        plot_for_mcus(n)


if __name__ == "__main__":
    main()
