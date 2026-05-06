"""Compare halo_on vs halo_off coordinator logs and save three separate plots.

Outputs (PNG, into --out-dir):
    halo_total_time.png    — per-layer end-to-end latency, OFF vs ON
    halo_breakdown.png     — compute vs net_oh, with net_oh % drop annotated
    halo_comm_volume.png   — send+recv stacked, OFF vs ON (mbv2_035_halo_comm style)

Usage:
    python test/plot_halo_comparison.py [--up-to blk4]
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# DEFAULT_OFF = Path(
#     "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_off/coordinator_block_info.log"
# )
# DEFAULT_ON = Path(
#     "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_on/coordinator_block_info.log"
# )
DEFAULT_OUT_DIR = Path(
    "/home/kyonqi/Project/RustProjects/Python_Sim_Infer/test/mbv2/halo/new_comparison_fix_nodelay"
)

DEFAULT_OFF = Path(
    "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_off/coordinator_block_phase_info.log"
)
DEFAULT_ON = Path(
    "/home/kyonqi/Project/RustProjects/logs/mbv2_035/fix_delay/4_mcus/halo_on/coordinator_block_phase_info.log"
)

# DEFAULT_OFF = Path(
#     "/home/kyonqi/Project/RustProjects/logs/mbv2/fix_delay/4_mcus/halo_off/coordinator_block_info.log"
# )
# DEFAULT_ON = Path(
#     "/home/kyonqi/Project/RustProjects/logs/mbv2/fix_delay/4_mcus/halo_on/coordinator_block_info.log"
# )



LAYER_RE = re.compile(
    r"Layer\s+(?P<idx>\S+)\s+\[\s*(?P<type>\S+)\s*\]\s+(?P<name>\S+):\s+"
    r"total=\s*(?P<total>[\d.]+)ms\s+"
    r"(?:send\(avg/max\)=\s*(?P<send_avg>[\d.]+)/\s*(?P<send_max>[\d.]+)ms\s+)?"
    r"cmpt\(avg/max\)=\s*(?P<cmpt_avg>[\d.]+)/\s*(?P<cmpt_max>[\d.]+)ms\s+"
    r"wait\(avg/max\)=\s*(?P<wait_avg>[\d.]+)/\s*(?P<wait_max>[\d.]+)ms\s+"
    r"net_oh\(avg/max\)=\s*(?P<oh_avg>[\d.]+)/\s*(?P<oh_max>[\d.]+)ms\s+"
    r"compress=\s*(?P<compress>[\d.]+)ms\s+"
    r"send=\s*(?P<send_actual>\S+)/\s*(?P<send_full>\S+)\s+\(save[^)]*\)\s+"
    r"recv=\s*(?P<recv_actual>\S+)/\s*(?P<recv_full>\S+)\s+\(save"
)

PHASE_RE = re.compile(
    r"engine_pre=\s*(?P<engine_pre>[\d.]+)\s+"
    r"build=\s*(?P<build>[\d.]+)\s+"
    r"send_phase=\s*(?P<send_phase>[\d.]+)\s+"
    r"recv_phase=\s*(?P<recv_phase>[\d.]+)\s+"
    r"engine_post=\s*(?P<engine_post>[\d.]+)"
)




def parse_bytes(s: str) -> float:
    s = s.strip()
    if s.endswith("MB"):
        return float(s[:-2]) * 1024 * 1024
    if s.endswith("KB"):
        return float(s[:-2]) * 1024
    if s.endswith("B"):
        return float(s[:-1])
    return float(s)


@dataclass
class LayerRow:
    idx: str
    type_: str
    name: str
    total_ms: float
    cmpt_avg_ms: float
    cmpt_max_ms: float
    wait_avg_ms: float
    wait_max_ms: float
    oh_avg_ms: float
    oh_max_ms: float
    send_actual: float
    recv_actual: float
    send_full: float
    recv_full: float
    # phase 字段
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
        return self.send_phase

    @property
    def comm_io(self) -> float:
        return max(0.0, self.wait_avg_ms - self.cmpt_avg_ms)

    @property
    def post_header_recv(self) -> float:
        return max(0.0, self.recv_phase - self.wait_avg_ms)


def parse_log(path: Path) -> list[LayerRow]:
    rows: list[LayerRow] = []
    last: LayerRow | None = None
    for line in path.read_text().splitlines():
        m = LAYER_RE.search(line)
        if m:
            d = m.groupdict()
            last = LayerRow(
                idx=d["idx"],
                type_=d["type"],
                name=d["name"],
                total_ms=float(d["total"]),
                cmpt_avg_ms=float(d["cmpt_avg"]),
                cmpt_max_ms=float(d["cmpt_max"]),
                wait_avg_ms=float(d["wait_avg"]),
                wait_max_ms=float(d["wait_max"]),
                oh_avg_ms=float(d["oh_avg"]),
                oh_max_ms=float(d["oh_max"]),
                send_actual=parse_bytes(d["send_actual"]),
                recv_actual=parse_bytes(d["recv_actual"]),
                send_full=parse_bytes(d["send_full"]),
                recv_full=parse_bytes(d["recv_full"]),
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
        raise ValueError(f"No layers parsed from {path} — check LAYER_RE")
    return rows



def short_label(r: LayerRow) -> str:
    if r.type_ == "BLOCK":
        base = r.name.split("_")[0]
        return f"{base}\n({r.idx})"
    return f"{r.name}\n({r.idx})"


# ─────────────────────────── plotting ────────────────────────────────


def plot_total_time_fig(off: list[LayerRow], on: list[LayerRow], out_path: Path) -> None:
    n = len(off)
    x = np.arange(n)
    w = 0.4
    off_t = np.array([r.total_ms for r in off])
    on_t = np.array([r.total_ms for r in on])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, off_t, w, label="halo OFF", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, on_t, w, label="halo ON", color="#2ca02c", alpha=0.85)

    y_max = max(off_t.max(), on_t.max())
    pad = y_max * 0.02
    for i in range(n):
        delta = off_t[i] - on_t[i]
        if abs(delta) >= 1.0:
            ax.annotate(
                f"-{delta:.1f}ms",
                xy=(i, max(off_t[i], on_t[i]) + pad),
                ha="center",
                fontsize=8,
                color="darkgreen" if delta > 0 else "red",
                fontweight="bold",
            )
    ax.set_ylim(top=y_max * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in off], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Layer total time (ms)")
    ax.set_title("Per-layer end-to-end latency: halo OFF vs ON", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


COLOR_COORD     = "#95a5a6"
COLOR_COMPUTE   = "#2980b9"
COLOR_NET       = "#e74c3c"
NET_OUTLINE     = "#641e16"
HATCH_SEND      = "..."
HATCH_COMM_IO   = ""
HATCH_POST      = "///"


def plot_breakdown_fig(off: list[LayerRow], on: list[LayerRow], out_path: Path) -> None:
    """Compute base + net_oh stacked, with net_oh split into 3 sub-phases via hatches."""
    n = len(off)
    x = np.arange(n)
    w = 0.4

    off_coord = np.array([r.coord_overhead   for r in off])
    on_coord  = np.array([r.coord_overhead   for r in on])
    off_cmpt = np.array([r.cmpt_avg_ms       for r in off])
    on_cmpt  = np.array([r.cmpt_avg_ms       for r in on])
    off_send = np.array([r.send_dispatch     for r in off])
    off_cio  = np.array([r.comm_io           for r in off])
    off_pst  = np.array([r.post_header_recv  for r in off])
    on_send  = np.array([r.send_dispatch     for r in on])
    on_cio   = np.array([r.comm_io           for r in on])
    on_pst   = np.array([r.post_header_recv  for r in on])
    off_oh   = off_send + off_cio + off_pst
    on_oh    = on_send  + on_cio  + on_pst

    HATCH_SEND = "..."     # send_dispatch
    HATCH_CIO  = ""        # comm + worker_io（主体留白）
    HATCH_POST = "///"     # post_header_recv

    plt.rcParams["hatch.color"]     = "white"
    plt.rcParams["hatch.linewidth"] = 1.0

    fig, ax = plt.subplots(figsize=(14, 6))

    # OFF (left, red net_oh)
    # ax.bar(x - w / 2, off_coord, w, color="#95a5a6",
    #        edgecolor="white", linewidth=0.5)                
    # bot = off_coord.copy()            
    ax.bar(x - w / 2, off_cmpt, w, color="#1f77b4")
    bot = off_cmpt.copy()
    ax.bar(x - w / 2, off_send, w, bottom=bot, color="#d62728",
           hatch=HATCH_SEND, edgecolor="white", linewidth=0.5)
    bot += off_send
    ax.bar(x - w / 2, off_cio, w, bottom=bot, color="#d62728",
           hatch=HATCH_CIO, edgecolor="white", linewidth=0.5)
    bot += off_cio
    ax.bar(x - w / 2, off_pst, w, bottom=bot, color="#d62728",
           hatch=HATCH_POST, edgecolor="white", linewidth=0.5)

    # ON (right, green net_oh)
    # ax.bar(x + w / 2, on_coord, w, color="#95a5a6",
    #        edgecolor="white", linewidth=0.5)
    # bot = on_coord.copy()
    ax.bar(x + w / 2, on_cmpt, w, color="#1f77b4")
    bot = on_cmpt.copy()
    ax.bar(x + w / 2, on_send, w, bottom=bot, color="#2ca02c",
           hatch=HATCH_SEND, edgecolor="white", linewidth=0.5)
    bot += on_send
    ax.bar(x + w / 2, on_cio, w, bottom=bot, color="#2ca02c",
           hatch=HATCH_CIO, edgecolor="white", linewidth=0.5)
    bot += on_cio
    ax.bar(x + w / 2, on_pst, w, bottom=bot, color="#2ca02c",
           hatch=HATCH_POST, edgecolor="white", linewidth=0.5)

    # # In-segment numbers for the 3 net_oh sub-phases (skip compute)
    # def annotate_net(pos, send_arr, cio_arr, pst_arr, cmpt_arr):
    #     for i in range(n):
    #         cur = cmpt_arr[i]                # 从 compute 顶部开始累加
    #         for arr in (send_arr, cio_arr, pst_arr):
    #             v = arr[i]
    #             if v >= 1.0:                 # 只有 ≥1ms 才标，避免挤
    #                 ax.text(pos[i], cur + v / 2, f"{v:.1f}",
    #                         ha="center", va="center",
    #                         fontsize=8, fontweight="bold", color="white")
    #             cur += v
    # annotate_net(x - w / 2, off_send, off_cio, off_pst, off_cmpt)
    # annotate_net(x + w / 2, on_send,  on_cio,  on_pst,  on_cmpt)


    # Per-layer net ↓N%
    off_total = off_coord + off_cmpt + off_oh
    on_total  = on_coord  + on_cmpt + on_oh
    y_max = max(off_total.max(), on_total.max())
    pad = y_max * 0.02
    for i in range(n):
        if off_oh[i] > 1e-3:
            pct = (off_oh[i] - on_oh[i]) / off_oh[i] * 100
            color = "darkgreen" if pct > 0 else "red"
            top = max(off_total[i], on_total[i])
            ax.text(x[i], top + pad, f"net ↓{pct:.0f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, color=color, fontweight="bold")
    ax.set_ylim(top=y_max * 1.18)

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in off], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        "Per-layer latency breakdown — left bar = halo OFF, right = halo ON",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # 整体 net_oh 下降百分比 → 放进 legend title
    total_off_oh = off_oh.sum()
    total_on_oh  = on_oh.sum()
    overall_pct = ((total_off_oh - total_on_oh) / total_off_oh * 100) if total_off_oh > 0 else 0.0

    from matplotlib.patches import Patch
    handles = [
        # Patch(facecolor="#95a5a6", label="coord_overhead"),          
        Patch(facecolor="#1f77b4", label="compute"),
        Patch(facecolor="#241f1f", label="net_oh  (halo OFF)"),
        Patch(facecolor="#2ca02c", label="net_oh  (halo ON)"),
        Patch(facecolor="#7f7f7f", hatch=HATCH_SEND, edgecolor="white",
              label="send_dispatch"),
        Patch(facecolor="#7f7f7f", hatch=HATCH_CIO,  edgecolor="white",
              label="comm + worker_io"),
        Patch(facecolor="#7f7f7f", hatch=HATCH_POST, edgecolor="white",
              label="post_header_recv"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right", ncol=2, fontsize=9, framealpha=0.95,
        title=f"Overall net_oh:  {total_off_oh:.1f} → {total_on_oh:.1f} ms   ↓{overall_pct:.1f}%",
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")




def plot_comm_volume_fig(off: list[LayerRow], on: list[LayerRow], out_path: Path) -> None:
    """Stacked send+recv per block, OFF/ON side by side. Layout: mbv2_035_halo_comm."""
    n = len(off)
    x = np.arange(n)
    width = 0.4

    send_off = np.array([r.send_actual / 1024 for r in off])
    recv_off = np.array([r.recv_actual / 1024 for r in off])
    send_on  = np.array([r.send_actual / 1024 for r in on])
    recv_on  = np.array([r.recv_actual / 1024 for r in on])
    off_total = send_off + recv_off
    on_total  = send_on + recv_on

    # Same red/green palette as the other two figures, send=darker / recv=lighter
    color_off = {"send": "#d62728", "recv": "#f4a3a4"}
    color_on  = {"send": "#2ca02c", "recv": "#a6dba0"}

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(x - width / 2, send_off, width,
           label="halo OFF: send", color=color_off["send"], alpha=0.85)
    ax.bar(x - width / 2, recv_off, width, bottom=send_off,
           label="halo OFF: recv", color=color_off["recv"], alpha=0.85)

    ax.bar(x + width / 2, send_on, width,
           label="halo ON: send", color=color_on["send"], alpha=0.85)
    ax.bar(x + width / 2, recv_on, width, bottom=send_on,
           label="halo ON: recv", color=color_on["recv"], alpha=0.85)

    y_max = off_total.max() if off_total.max() > 0 else 1.0
    pad = y_max * 0.02
    for i in range(n):
        if off_total[i] > 0:
            save = (off_total[i] - on_total[i]) / off_total[i] * 100
            if abs(save) > 1:
                top = max(off_total[i], on_total[i])
                ax.text(x[i], top + pad, f"↓{save:.0f}%",
                        ha="center", va="bottom", fontsize=8.5,
                        color="darkgreen", fontweight="bold")
    ax.set_ylim(top=y_max * 1.22)

    tot_off = off_total.sum()
    tot_on = on_total.sum()
    send_off_sum, send_on_sum = send_off.sum(), send_on.sum()
    recv_off_sum, recv_on_sum = recv_off.sum(), recv_on.sum()
    tot_save = (tot_off - tot_on) / tot_off * 100 if tot_off > 0 else 0
    send_save = (send_off_sum - send_on_sum) / send_off_sum * 100 if send_off_sum > 0 else 0
    recv_save = (recv_off_sum - recv_on_sum) / recv_off_sum * 100 if recv_off_sum > 0 else 0

    summary = (
        f"Total Comm. Volume (send + recv):\n"
        f"  Halo-off : {tot_off:>8.2f} KB\n"
        f"  Halo-on  : {tot_on:>8.2f} KB\n"
        f"  Saved    : {tot_off - tot_on:>8.2f} KB  ({tot_save:+.1f}%)\n"
        f"  ── send  : {send_off_sum:6.1f} → {send_on_sum:6.1f} KB  ({send_save:+.1f}%)\n"
        f"  ── recv  : {recv_off_sum:6.1f} → {recv_on_sum:6.1f} KB  ({recv_save:+.1f}%)"
    )
    ax.text(0.985, 0.97, summary, transform=ax.transAxes, fontsize=10.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdfefe",
                      alpha=0.95, edgecolor="#bdc3c7"),
            family="monospace")

    ax.set_xlabel("Block Partition (Layer Range)", fontsize=11)
    ax.set_ylabel("Communication Volume (KB)", fontsize=11)
    ax.set_title(
        "Per-Block Comm. Volume — Halo OFF vs ON (4 MCUs, MBV2-0.35)",
        fontsize=13, fontweight="bold", pad=38,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in off],
                       rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
        ncol=4, fontsize=10.5, frameon=True, framealpha=0.95,
        edgecolor="#bdc3c7",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")



def print_summary(off: list[LayerRow], on: list[LayerRow]) -> None:
    tot_off = sum(r.total_ms for r in off)
    tot_on = sum(r.total_ms for r in on)
    send_off = sum(r.send_actual for r in off)
    send_on = sum(r.send_actual for r in on)
    recv_off = sum(r.recv_actual for r in off)
    recv_on = sum(r.recv_actual for r in on)
    print(f"Total time: OFF={tot_off:.2f}ms  ON={tot_on:.2f}ms  saved={tot_off-tot_on:.2f}ms ({100*(tot_off-tot_on)/tot_off:.1f}%)")
    print(f"Total send: OFF={send_off/1024:.2f}KB  ON={send_on/1024:.2f}KB  saved={100*(send_off-send_on)/max(send_off,1):.1f}%")
    print(f"Total recv: OFF={recv_off/1024:.2f}KB  ON={recv_on/1024:.2f}KB  saved={100*(recv_off-recv_on)/max(recv_off,1):.1f}%")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--off", type=Path, default=DEFAULT_OFF)
    p.add_argument("--on", type=Path, default=DEFAULT_ON)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--up-to", type=str, default="blk4",
                   help="Truncate at and including this block name, e.g. 'blk4'")
    args = p.parse_args()

    off = parse_log(args.off)
    on = parse_log(args.on)
    if len(off) != len(on):
        raise ValueError(f"Layer count mismatch: OFF={len(off)} vs ON={len(on)}")
    if [r.idx for r in off] != [r.idx for r in on]:
        raise ValueError("Layer indices don't line up between OFF and ON logs")

    if args.up_to:
        cutoff = None
        for i, r in enumerate(off):
            if args.up_to in r.name:
                cutoff = i + 1
        if cutoff is None:
            raise ValueError(f"--up-to '{args.up_to}' did not match any layer name")
        off = off[:cutoff]
        on = on[:cutoff]
        print(f"Truncated to first {cutoff} entries (up to and including '{args.up_to}')")

    print_summary(off, on)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_total_time_fig(off, on, args.out_dir / "mbv2_halo_total_time.png")
    plot_breakdown_fig(off, on, args.out_dir / "mbv2_halo_breakdown.png")
    plot_comm_volume_fig(off, on, args.out_dir / "mbv2_halo_comm_volume.png")


if __name__ == "__main__":
    main()




# DEFAULT_OFF = Path(
#     "/home/kyonqi/Project/RustProjects/logs/mbv2/fix_delay/4_mcus/halo_off/coordinator_block_info.log"
# )
# DEFAULT_ON = Path(
#     "/home/kyonqi/Project/RustProjects/logs/mbv2/fix_delay/4_mcus/halo_on/coordinator_block_info.log"
# )