#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMLU_MODE="${MMLU_MODE:-diagnostic}"
case "$MMLU_MODE" in
  diagnostic|full_depth) ;;
  *) echo "MMLU_MODE must be diagnostic or full_depth" >&2; exit 2 ;;
esac
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  mkdir -p "$BASE_DIR/logs/benchmark/mmlu/$MMLU_MODE"
  OUTPUT_DIR="$(mktemp -d "$BASE_DIR/logs/benchmark/mmlu/$MMLU_MODE/run-XXXXXX")"
fi
export OUTPUT_DIR
export QUESTIONS_PATH="${QUESTIONS_PATH:-$BASE_DIR/data/raw_benchmarks/mmlu_en.jsonl}"

bash "$BASE_DIR/scripts/run_mmlu_benchmark.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found/executable: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" "$BASE_DIR/scripts/aggregate_mmlu_results.py" \
  --logs "$OUTPUT_DIR" \
  --dataset "$QUESTIONS_PATH" \
  --output "$OUTPUT_DIR/mmlu_summary.csv"
