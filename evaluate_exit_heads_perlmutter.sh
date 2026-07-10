#!/usr/bin/env bash
set -euo pipefail

module load conda
# Initialize conda in non-interactive shells and activate env
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh" || true
conda activate ddl || true

# Log file setup
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/evaluate_exit_heads_${TS}.log"
echo "[log] Writing to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

USERNAME="${USER}"
readonly USERNAME

export HF_HOME="${HF_HOME:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
# Optional GPU selection
if [[ -n "${GPU_ID:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    DEVICE="${DEVICE:-cuda:0}"
  else
    DEVICE="${DEVICE:-cuda:${GPU_ID}}"
  fi
fi
DATASET_PATH="${DATASET_PATH:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/extracted/exit-model/data/prepared_dataset/calibration/oasst1_llama31.jsonl}"
METADATA="${METADATA:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/extracted/exit-model/artifacts/trained-exit-heads_kd_round_final/metadata.json}"
THRESHOLDS="${THRESHOLDS:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/extracted/exit-model/artifacts/trained-exit-heads_kd_round_final/tr_0.4/thresholds.json}"
LIMIT_OPT=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_OPT=(--limit "${LIMIT}")
fi

# Verbosity and progress
VERBOSE_OPT=(--verbose)
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50}"

echo "[runbook] HF_HOME      : ${HF_HOME}"
echo "[runbook] Dataset path : ${DATASET_PATH}"
echo "[runbook] Metadata     : ${METADATA}"
echo "[runbook] Thresholds   : ${THRESHOLDS}"
echo "[runbook] Device       : ${DEVICE}"
if [[ -n "${GPU_ID:-}" ]]; then
  echo "[runbook] GPU_ID/CUDA_VISIBLE_DEVICES : ${GPU_ID}/${CUDA_VISIBLE_DEVICES:-unset}"
fi
if [[ -n "${LIMIT:-}" ]]; then
  echo "[runbook] Limit        : ${LIMIT}"
fi

python -m training.evaluate_exit_heads \
  --manifest config/manifest.json \
  --metadata "${METADATA}" \
  --thresholds "${THRESHOLDS}" \
  --dataset "${DATASET_PATH}" \
  --device "${DEVICE}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  "${VERBOSE_OPT[@]}" \
  "${LIMIT_OPT[@]}" \
  "$@"
