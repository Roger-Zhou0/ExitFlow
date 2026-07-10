#!/usr/bin/env python3
"""Plot GPU utilization curves from logs/HIT_inference/<timestamp>/gpu_log.csv.

Usage:
  python scripts/plot_hit_inference.py --log-root logs/HIT_inference
  python scripts/plot_hit_inference.py --run-dir logs/HIT_inference/20241128_120000
  python scripts/plot_hit_inference.py --run-dir logs/HIT_inference/20241128_120000 --show

Outputs:
  - gpu_util.png : GPU 核心利用率曲线
  - gpu_mem.png  : 显存占用曲线
  保存到指定 run 目录；若 --show 则弹出窗口。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_run(log_root: Path) -> Path:
    candidates = [p for p in log_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {log_root}")
    return sorted(candidates)[-1]


def load_gpu_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"gpu log not found: {path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns or "gpu" not in df.columns:
        raise ValueError(f"gpu log missing required columns: {df.columns}")
    t0 = df["ts"].min()
    df["t_rel"] = df["ts"] - t0
    return df


def plot_series(df: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: Path, show: bool) -> None:
    plt.figure(figsize=(10, 5))
    for gpu_id in sorted(df["gpu"].unique()):
        sub = df[df["gpu"] == gpu_id]
        plt.plot(sub["t_rel"], sub[value_col], label=f"GPU{gpu_id}")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GPU utilization from HIT_inference logs")
    parser.add_argument("--log-root", default="logs/HIT_inference", help="Root directory containing timestamped runs")
    parser.add_argument("--run-dir", help="Specific run directory (if not provided, pick latest under log-root)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    log_root = Path(args.log_root)
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(log_root)
    gpu_csv = run_dir / "gpu_log.csv"

    df = load_gpu_log(gpu_csv)
    util_path = run_dir / "gpu_util.png"
    mem_path = run_dir / "gpu_mem.png"

    plot_series(df, "gpu_util", "GPU Util (%)", "GPU Utilization over Time", util_path, args.show)
    plot_series(df, "mem_used_mb", "Memory Used (MB)", "GPU Memory over Time", mem_path, args.show)

    print(f"[plot] wrote {util_path}")
    print(f"[plot] wrote {mem_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
