from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 颜色与样式配置
COLOR_COORD   = "#95a5a6"
COLOR_COMPUTE = "#2980b9"
COLOR_NET     = "#e74c3c"
NET_OUTLINE   = "#641e16"

# Hatch 样式
HATCH_SEND    = "..."
HATCH_COMM_IO = ""
HATCH_POST    = "///"

DEFAULT_BEFORE = Path("/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_off/bugs_in_init/coordinator_block_phase_info.log")
DEFAULT_AFTER = Path("/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_off/coordinator_block_phase_info.log")
DEFAULT_OUT = Path("/home/kyonqi/Project/RustProjects/Python_Sim_Infer/test/mbv2/halo/halo_off_phase_breakdown_compare.png")


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
    wait_avg: float = 0.0
    engine_pre: float = 0.0
    build: float = 0.0
    send_phase: float = 0.0
    recv_phase: float = 0.0
    engine_post: float = 0.0

    @property
    def coord_overhead(self) -> float:
        return self.engine_pre + self.build + self.engine_post

    @property
    def send_dispatch(self) -> float:
        """coord 端把 task 塞进内核 buffer 的时间"""
        return self.send_phase

    @property
    def comm_io(self) -> float:
        """forward wire + worker recv/pre/post IO + return header wire"""
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
    
    if not path.exists():
        print(f"Warning: {path} not found. (Mocking data for test if needed)")
        return rows

    for line in path.read_text().splitlines():
        if h := HEADER_RE.search(line):
            last = Row(
                idx=h["idx"], type_=h["type"], name=h["name"],
                total_ms=float(h["total"]),
                cmpt_avg=float(h["cmpt_avg"]),
                wait_avg=float(h["wait_avg"]),
            )
            rows.append(last)
            continue
        
        if last and (p := PHASE_RE.search(line)):
            last.engine_pre  = float(p["engine_pre"])
            last.build       = float(p["build"])
            last.send_phase  = float(p["send_phase"])
            last.recv_phase  = float(p["recv_phase"])
            last.engine_post = float(p["engine_post"])
            
    if not rows:
        raise ValueError(f"No layers parsed from {path}")
    return rows


def short_label(r: Row) -> str:
    return r.name.split("_")[0] if r.type_ == "BLOCK" else r.name


def plot(before: list[Row], after: list[Row], out_path: Path) -> None:
    n = len(before)
    x = np.arange(n)
    w = 0.38
    gap = 0.02

    # 优化：提取通用的数据结构转换逻辑
    def extract_metrics(rows: list[Row]) -> dict[str, np.ndarray]:
        return {
            "cmpt":    np.array([r.cmpt_avg for r in rows]),
            "coord":   np.array([r.coord_overhead for r in rows]),
            "send":    np.array([r.send_dispatch for r in rows]),
            "comm_io": np.array([r.comm_io for r in rows]),
            "post":    np.array([r.post_header_recv for r in rows]),
            "net":     np.array([r.net_overhead for r in rows]),
            "tot":     np.array([r.total_ms for r in rows]),
        }

    bef = extract_metrics(before)
    aft = extract_metrics(after)

    plt.rcParams["hatch.color"]     = NET_OUTLINE
    plt.rcParams["hatch.linewidth"] = 1.0

    fig, ax = plt.subplots(figsize=(14, 7.5))

    pos_b = x - w / 2 - gap
    pos_a = x + w / 2 + gap

    def stack_bar(pos, vals, attach_legend):
        segs = [
            (vals["coord"],   "coord overhead",                           COLOR_COORD,   ""),
            (vals["cmpt"],    "compute",                                  COLOR_COMPUTE, ""),
            (vals["send"],    "net overhead - send data",                 COLOR_NET,     HATCH_SEND),
            (vals["comm_io"], "net overhead - comm header + worker io",   COLOR_NET,     HATCH_COMM_IO),
            (vals["post"],    "net overhead - recv data",                 COLOR_NET,     HATCH_POST),
        ]

        bottom = np.zeros(n)
        for arr, label, color, hatch in segs:
            # 修复：只有 attach_legend=True 时才传递 label，避免图例重复
            lbl = label if attach_legend else None
            
            # 修复：将 x 替换为 pos，解决柱状图重叠问题
            ax.bar(pos, arr, w, bottom=bottom, label=lbl, color=color,
                   edgecolor="white", linewidth=0.6, hatch=hatch, zorder=2)
            
            for i, val in enumerate(arr):
                if val >= 1.0:
                    # 修复：文本 X 轴坐标必须使用 pos[i] 而非 i
                    ax.text(
                        pos[i], bottom[i] + val / 2, f"{val:.1f}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                        zorder=3,
                    )
            bottom += arr

    stack_bar(pos_b, bef, attach_legend=True)
    stack_bar(pos_a, aft, attach_legend=False)

    y_max = max(bef["tot"].max(), aft["tot"].max())
    pad = y_max * 0.013
    
    for i in range(n):
        ax.text(pos_b[i], bef["tot"][i] + pad, f"{bef['tot'][i]:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1b2631")
        ax.text(pos_a[i], aft["tot"][i] + pad, f"{aft['tot'][i]:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1b2631")
        
        delta = bef["tot"][i] - aft["tot"][i]
        if abs(delta) >= 0.5:
            ax.annotate(
                f"−{delta:.1f} ms" if delta > 0 else f"+{-delta:.1f} ms",
                xy=(x[i], max(bef["tot"][i], aft["tot"][i]) + pad * 4),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="darkgreen" if delta > 0 else "darkred",
            )
            
    ax.set_ylim(top=y_max * 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in before], fontsize=11)
    ax.tick_params(axis="x", pad=18)
    
    for i in range(n):
        ax.text(pos_b[i], -y_max * 0.018, "before",
                ha="center", va="top", fontsize=8.5, color="#7f8c8d")
        ax.text(pos_a[i], -y_max * 0.018, "after",
                ha="center", va="top", fontsize=8.5, color="#7f8c8d")

    ax.set_ylabel("Time (ms)")
    ax.set_title(
        "Per-layer Latency Breakdown: Before vs After Disabling ACK Delay\n"
        "halo_off · MBV2-0.35 · 4 MCUs",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95,
              title="5-segment breakdown ↓", title_fontsize=9.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_summary(before: list[Row], after: list[Row]) -> None:
    tot_b = sum(r.total_ms for r in before)
    tot_a = sum(r.total_ms for r in after)
    saved = tot_b - tot_a
    print(f"Total: BEFORE={tot_b:.2f}ms  AFTER={tot_a:.2f}ms  "
          f"saved={saved:.2f}ms ({100*saved/tot_b:.1f}%)\n")
          
    print(f"{'layer':>14s}  {'tot_b':>7s}  {'tot_a':>7s}  "
          f"{'cmpt_b':>7s}  {'cmpt_a':>7s}  "
          f"{'comm_b':>7s}  {'comm_a':>7s}  "
          f"{'post_b':>7s}  {'post_a':>7s}  {'Δms':>7s}")
          
    for b, a in zip(before, after):
        d = b.total_ms - a.total_ms
        # 修复：使用了正确的属性名（cmpt_avg, comm_io, post_header_recv）
        print(f"{a.name[:14]:>14s}  "
              f"{b.total_ms:>7.2f}  {a.total_ms:>7.2f}  "
              f"{b.cmpt_avg:>7.2f}  {a.cmpt_avg:>7.2f}  "
              f"{b.comm_io:>7.2f}  {a.comm_io:>7.2f}  "
              f"{b.post_header_recv:>7.2f}  {a.post_header_recv:>7.2f}  {d:>+7.2f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    p.add_argument("--after",  type=Path, default=DEFAULT_AFTER)
    p.add_argument("--out",    type=Path, default=DEFAULT_OUT)
    p.add_argument("--up-to",  type=str,  default="blk4")
    args = p.parse_args()

    try:
        before = parse_log(args.before)
        after = parse_log(args.after)
    except ValueError as e:
        print(f"Error parsing logs: {e}")
        return

    if [r.idx for r in before] != [r.idx for r in after]:
        raise ValueError("Layer indices don't match between BEFORE and AFTER")

    if args.up_to:
        cutoff = next((i + 1 for i, r in enumerate(before) if args.up_to in r.name), None)
        if cutoff is None:
            raise ValueError(f"--up-to '{args.up_to}' not found")
        before = before[:cutoff]
        after = after[:cutoff]

    print_summary(before, after)
    plot(before, after, args.out)


if __name__ == "__main__":
    main()