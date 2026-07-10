#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$BASE_DIR/logs/batch_results"

PYTHON_BIN="${PYTHON_BIN:-/global/homes/x/xz446/.conda/envs/ddl/bin/python3.10}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || command -v python3 || command -v python)}"
    if [[ -z "$PYTHON_BIN" ]]; then
      echo "python interpreter not found" >&2
      exit 1
    fi
  fi
fi

$PYTHON_BIN "$BASE_DIR/runtime/autoregressive_batch.py" \
  --manifest "$BASE_DIR/config/manifest.json" \
  --metadata "$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/metadata.json" \
  --thresholds "$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/oasst1/tr_0.8/thresholds.json" \
  --questions "$BASE_DIR/experiments/questions_batch.txt" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --device cuda \
  --show-steps
