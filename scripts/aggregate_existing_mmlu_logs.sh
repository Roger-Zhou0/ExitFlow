#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${1:-$BASE_DIR/logs/benchmark/mmlu}"
DATASET_PATH="${2:-$BASE_DIR/data/raw_benchmarks/mmlu_en.jsonl}"
OUTPUT_PATH="${3:-$LOG_DIR/mmlu_summary.csv}"

python3 "$BASE_DIR/scripts/aggregate_mmlu_results.py" \
  --logs "$LOG_DIR" \
  --dataset "$DATASET_PATH" \
  --output "$OUTPUT_PATH"
