#!/usr/bin/env python3
"""Create a presentation-ready GPU usage plot for a HIT_inference run.

Usage:
  python scripts/plot_hit_inference_presentation.py --run-dir logs/HIT_inference/20251128_081119
  python scripts/plot_hit_inference_presentation.py --log-root logs/HIT_inference  # picks latest run

Outputs:
  - gpu_log_presentation.png in the target run directory (overridable via --out-name)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a presentation-style GPU usage chart.")
    parser.add_argument(
        "--log-root",
        default="logs/HIT_inference",
        help="Root directory containing timestamped HIT_inference runs.",
    )
    parser.add_argument(
        "--run-dir",
        help="Specific run directory. If omitted, the latest under --log-root is used.",
    )
    parser.add_argument(
        "--out-name",
        default="gpu_log_presentation.png",
        help="Output filename placed inside the run directory.",
    )
    parser.add_argument(
        "--mplconfig",
        default=".mplconfig",
        help="Writable directory for Matplotlib cache (created if missing).",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    return parser.parse_args(list(argv))


def find_latest_run(log_root: Path) -> Path:
    candidates = [p for p in log_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {log_root}")
    return sorted(candidates)[-1]


def load_gpu_log(path: Path) -> Tuple[List[dict], float, float]:
    import csv

    if not path.exists():
        raise FileNotFoundError(f"gpu log not found: {path}")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"gpu log is empty: {path}")
    start_ts = float(rows[0]["ts"])
    end_ts = float(rows[-1]["ts"])
    return rows, start_ts, end_ts


def ensure_mplconfig(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path))


def plot_presentation(rows: List[dict], start_ts: float, end_ts: float, out_path: Path, show: bool) -> None:
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    duration = end_ts - start_ts

    by_gpu = {}
    total_gpu_util = 0.0
    total_mem_util = 0.0
    total_count = 0
    for row in rows:
        gpu = int(row["gpu"])
        t_rel = float(row["ts"]) - start_ts
        gpu_util = float(row["gpu_util"])
        mem_util = float(row["mem_util"])
        mem_used = float(row["mem_used_mb"])
        mem_total = float(row["mem_total_mb"])
        by_gpu.setdefault(
            gpu,
            {"t": [], "gpu_util": [], "mem_util": [], "mem_used": [], "mem_total": mem_total},
        )
        g = by_gpu[gpu]
        g["t"].append(t_rel)
        g["gpu_util"].append(gpu_util)
        g["mem_util"].append(mem_util)
        g["mem_used"].append(mem_used)
        total_gpu_util += gpu_util
        total_mem_util += mem_util
        total_count += 1

    avg_gpu_util = total_gpu_util / total_count if total_count else 0.0
    avg_mem_util = total_mem_util / total_count if total_count else 0.0

    palette = mpl.colormaps.get_cmap("viridis", len(by_gpu))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    fig.patch.set_facecolor("#0e1117")

    for ax in axes:
        ax.set_facecolor("#11151c")
        ax.grid(color="#30363d", alpha=0.6, linestyle="--", linewidth=0.8)
        ax.tick_params(colors="#d8dee9", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    for idx, (gpu, series) in enumerate(sorted(by_gpu.items())):
        color = palette(idx)
        label = f"GPU {gpu}"
        axes[0].plot(series["t"], series["gpu_util"], label=label, color=color, linewidth=2)
        axes[0].fill_between(series["t"], series["gpu_util"], alpha=0.08, color=color)
        axes[1].plot(series["t"], series["mem_util"], label=label, color=color, linewidth=2)
        axes[1].fill_between(series["t"], series["mem_util"], alpha=0.08, color=color)
        axes[2].plot(series["t"], series["mem_used"], label=label, color=color, linewidth=2)
        axes[2].fill_between(series["t"], series["mem_used"], alpha=0.08, color=color)

    # Average utilization lines
    axes[0].axhline(avg_gpu_util, color="#e5e9f0", linestyle=":", linewidth=1.5, label=f"Avg {avg_gpu_util:.1f}%")
    axes[1].axhline(avg_mem_util, color="#e5e9f0", linestyle=":", linewidth=1.5, label=f"Avg {avg_mem_util:.1f}%")

    axes[0].set_ylabel("GPU Util (%)", color="#d8dee9", fontsize=11)
    axes[1].set_ylabel("Mem Util (%)", color="#d8dee9", fontsize=11)
    axes[2].set_ylabel("Mem Used (MB)", color="#d8dee9", fontsize=11)
    axes[2].set_xlabel("Time since start (s)", color="#d8dee9", fontsize=11)

    subtitle = f"Start {start_dt.isoformat()}  |  Duration {duration:.1f}s  |  Avg GPU {avg_gpu_util:.1f}%  |  Avg Mem {avg_mem_util:.1f}%"
    axes[0].set_title("GPU Usage Overview", loc="left", color="#e5e9f0", fontsize=16, pad=10)
    axes[0].text(
        0.99,
        1.1,
        subtitle,
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        color="#a5adcb",
        fontsize=10,
    )

    for ax in axes:
        ax.legend(frameon=False, loc="upper right", labelcolor="#d8dee9")
        ax.set_xlim(left=0)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    log_root = Path(args.log_root)
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(log_root)
    gpu_csv = run_dir / "gpu_log.csv"

    ensure_mplconfig(Path(args.mplconfig))
    rows, start_ts, end_ts = load_gpu_log(gpu_csv)
    out_path = run_dir / args.out_name

    plot_presentation(rows, start_ts, end_ts, out_path, args.show)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
