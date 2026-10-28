#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# An explicit interpreter is respected. Otherwise, use a working interpreter
# from PATH rather than an old, machine-specific Conda location.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found/executable: $PYTHON_BIN" >&2
  exit 1
fi

MANIFEST_PATH="${MANIFEST_PATH:-$BASE_DIR/config/manifest.json}"
METADATA_PATH="${METADATA_PATH:-$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/metadata.json}"
THRESHOLDS_PATH="${THRESHOLDS_PATH:-$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/dolly_15k/tr_0.6/thresholds.json}"
QUESTIONS_PATH="${QUESTIONS_PATH:-$BASE_DIR/data/raw_benchmarks/mmlu_en.jsonl}"

# Both modes execute the *entire* backbone at every generation step.
# Diagnostic uses intermediate head decisions; full_depth uses the final head.
# Neither mode measures physical layer skipping.
MMLU_MODE="${MMLU_MODE:-diagnostic}"
case "$MMLU_MODE" in
  diagnostic)
    DEFAULT_OUTPUT_DIR="$BASE_DIR/logs/benchmark/mmlu/diagnostic"
    ;;
  full_depth)
    set -- --force-final-exit "$@"
    DEFAULT_OUTPUT_DIR="$BASE_DIR/logs/benchmark/mmlu/full_depth"
    ;;
  *)
    echo "MMLU_MODE must be diagnostic or full_depth (got '$MMLU_MODE')" >&2
    exit 2
    ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

for required in "$MANIFEST_PATH" "$METADATA_PATH" "$THRESHOLDS_PATH" "$QUESTIONS_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
if find "$OUTPUT_DIR" -maxdepth 1 -name 'question_*.json' -print -quit | grep -q .; then
  echo "Refusing to mix new predictions with existing logs in: $OUTPUT_DIR" >&2
  echo "Use a fresh OUTPUT_DIR for each run." >&2
  exit 1
fi
echo "ExitFlow MMLU full-backbone diagnostic: $MMLU_MODE; output: $OUTPUT_DIR"
"$PYTHON_BIN" "$BASE_DIR/runtime/autoregressive_batch.py" \
  --manifest "$MANIFEST_PATH" \
  --metadata "$METADATA_PATH" \
  --thresholds "$THRESHOLDS_PATH" \
  --questions "$QUESTIONS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --seed "${SEED:-0}" \
  --device "${DEVICE:-cuda}" \
  "$@"
