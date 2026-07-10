#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export BASE_DIR
LOG_DIR="$BASE_DIR/logs/benchmark/mmlu"
SUMMARY_CSV="$LOG_DIR/mmlu_summary.csv"

bash "$BASE_DIR/scripts/run_mmlu_benchmark.sh"

python3 <<'PY'
import subprocess, sys
from pathlib import Path

import os

BASE_DIR = Path(os.environ["BASE_DIR"])
LOG_DIR = BASE_DIR / "logs" / "benchmark" / "mmlu"
SUMMARY = LOG_DIR / "mmlu_summary.csv"

cmd = [
    str(BASE_DIR / "scripts" / "aggregate_mmlu_results.py"),
    "--logs",
    str(LOG_DIR),
    "--dataset",
    str(BASE_DIR / "data" / "raw_benchmarks" / "mmlu_en.jsonl"),
    "--output",
    str(SUMMARY),
]
subprocess.run(cmd, check=True)
PY
