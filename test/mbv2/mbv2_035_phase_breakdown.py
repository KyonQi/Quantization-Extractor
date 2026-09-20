"""Per-layer latency decomposition into 3 honest categories.

   total = coord_overhead + compute + net_overhead

where
   coord_overhead = engine_pre + build + engine_post   (pure Python/numpy on coord)
   compute        = avg over workers of MCU compute    (the productive work)
   net_overhead   = everything else (forward wire + worker IO + return wire + parse)

The current measurement can't cleanly separate forward / worker_io / return inside
net_overhead — that requires worker-side timestamps in ResultMessage.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle    # ← 新增

# coord 灰、compute 冷蓝、net_overhead 整体一种红
COLOR_COORD   = "#95a5a6"
COLOR_COMPUTE = "#2980b9"
COLOR_NET     = "#e74c3c"     # ← net_overhead 全部一种红
NET_OUTLINE   = "#641e16"     # 外框 / hatch 颜色

# 用 hatch 区分 net 内部 3 段
HATCH_SEND      = "..."       # 点 — send_dispatch (最薄)
HATCH_COMM_IO   = ""          # 主体留白（最大块，最容易读数字）
HATCH_POST      = "///"       # 斜线 — post_header_recv


DEFAULT_LOG = Path(
    "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_off/bugs_in_init/coordinator_block_phase_info.log"
)
DEFAULT_OUT = Path(
    "/home/kyonqi/Project/RustProjects/Python_Sim_Infer/test/mbv2/halo/halo_off_phase_breakdown.png"
)


HEADER_RE = re.compile(
    r"Layer\s+(?P<idx>\S+)\s+\[\s*(?P<type>\S+)\s*\]\s+(?P<name>\S+):\s+"
    r"total=\s*(?P<total>[\d.]+)ms\s+"
    r"send\(avg/max\)=\s*(?P<send_avg>[\d.]+)/\s*[\d.]+ms\s+"
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
    cmpt_avg: float = 0.0
    wait_avg: float = 0.0          # ← 新增
    engine_pre: float = 0.0
    build: float = 0.0
    send_phase: float = 0.0
    recv_phase: float = 0.0
    engine_post: float = 0.0

    @property
    def coord_overhead(self) -> float:
        return self.engine_pre + self.build + self.engine_post

    # ── net_overhead 拆成 3 段 ──
    @property
    def send_dispatch(self) -> float:
        """coord 端把 task 塞进内核 buffer 的时间"""
        return self.send_phase

    @property
    def comm_io(self) -> float:
        """forward wire + worker recv/pre/post IO + return header wire（无法干净拆）"""
        return max(0.0, self.wait_avg - self.cmpt_avg)

    @property
    def post_header_recv(self) -> float:
        """body wire + parse + load imbalance"""
        return max(0.0, self.recv_phase - self.wait_avg)

    @property
    def net_overhead(self) -> float:
        return self.send_dispatch + self.comm_io + self.post_header_recv



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
            last.engine_pre  = float(p["engine_pre"])
            last.build       = float(p["build"])
            last.send_phase  = float(p["send_phase"])
            last.recv_phase  = float(p["recv_phase"])
            last.engine_post = float(p["engine_post"])
    if not rows:
        raise ValueError(f"No layers parsed from {path}")
    return rows


def short_label(r: Row) -> str:
    if r.type_ == "BLOCK":
        return r.name.split("_")[0]
    return r.name


def plot(rows: list[Row], out_path: Path) -> None:
    n = len(rows)
    x = np.arange(n)
    w = 0.6

    # 让 hatch 的线条用深红色（默认是黑色），跟红色底搭配协调
    plt.rcParams["hatch.color"]     = NET_OUTLINE
    plt.rcParams["hatch.linewidth"] = 1.1

    coord = np.array([r.coord_overhead   for r in rows])
    cmpt  = np.array([r.cmpt_avg         for r in rows])
    snd   = np.array([r.send_dispatch    for r in rows])
    cio   = np.array([r.comm_io          for r in rows])
    pst   = np.array([r.post_header_recv for r in rows])
    net   = snd + cio + pst
    total = np.array([r.total_ms         for r in rows])

    fig, ax = plt.subplots(figsize=(13, 7))

    # (arr, label, color, hatch)
    segs = [
        (coord, "coord overhead",       COLOR_COORD,   ""),
        (cmpt,  "compute",              COLOR_COMPUTE, ""),
        (snd,   "net overhead - send data",    COLOR_NET,     HATCH_SEND),
        (cio,   "net overhead - comm header + worker io", COLOR_NET,     HATCH_COMM_IO),
        (pst,   "net overhead - recv data", COLOR_NET,     HATCH_POST),
    ]

    bottom = np.zeros(n)
    for arr, label, color, hatch in segs:
        ax.bar(x, arr, w, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.6, hatch=hatch, zorder=2)
        for i, val in enumerate(arr):
            if val >= 1.0:
                ax.text(
                    i, bottom[i] + val / 2, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    zorder=3,
                )
        bottom += arr

    # 在整个 net_overhead 区域外加一圈深红实线，强化"这是一个块"
    for i in range(n):
        y_bot = coord[i] + cmpt[i]
        h = net[i]
        if h > 0:
            ax.add_patch(Rectangle(
                (i - w / 2 - 0.005, y_bot),
                w + 0.01,
                h,
                fill=False,
                edgecolor=NET_OUTLINE,
                linewidth=1.6,
                zorder=4,
            ))
            ax.annotate(
                f"net oh\n{h:.1f}ms",
                xy=(i + w / 2 + 0.02, y_bot + h / 2),
                xytext=(8, 0), textcoords="offset points",
                ha="left", va="center",
                fontsize=8.5, color=NET_OUTLINE, fontweight="bold",
            )

    # Total label on top
    y_max = total.max()
    for i, t in enumerate(total):
        ax.text(i, t + y_max * 0.012, f"{t:.1f} ms",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1b2631")
    ax.set_ylim(top=y_max * 1.13)

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in rows], fontsize=11)
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        "Per-layer Latency: coord / compute / net_overhead \n"
        "halo_off · DelayACK · MBV2-0.35 · 4 MCUs",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95,
              title="net_overhead breakdown ↓", title_fontsize=9.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log",   type=Path, default=DEFAULT_LOG)
    p.add_argument("--out",   type=Path, default=DEFAULT_OUT)
    p.add_argument("--up-to", type=str,  default="blk4")
    args = p.parse_args()

    rows = parse_log(args.log)
    if args.up_to:
        cutoff = next((i + 1 for i, r in enumerate(rows) if args.up_to in r.name), None)
        if cutoff is None:
            raise ValueError(f"--up-to '{args.up_to}' not found")
        rows = rows[:cutoff]

    plot(rows, args.out)


if __name__ == "__main__":
    main()
