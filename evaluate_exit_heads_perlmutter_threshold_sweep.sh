#!/usr/bin/env bash
set -euo pipefail

module load conda
# Initialize conda in non-interactive shells and activate env
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh" || true
conda activate ddl || true

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

USERNAME="${USER}"
readonly USERNAME

GPU_ARG=""
declare -a THRESHOLD_FILTER=()
SCRIPT_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --gpu-id)
      if (( $# < 2 )); then
        echo "[error] --gpu-id requires a value" >&2
        exit 1
      fi
      GPU_ARG="$2"
      shift 2
      ;;
    --gpu-id=*)
      GPU_ARG="${1#*=}"
      shift
      ;;
    --threshold)
      if (( $# < 2 )); then
        echo "[error] --threshold requires a value" >&2
        exit 1
      fi
      THRESHOLD_FILTER+=("$2")
      shift 2
      ;;
    --threshold=*)
      THRESHOLD_FILTER+=("${1#*=}")
      shift
      ;;
    --)
      shift
      SCRIPT_ARGS+=("$@")
      break
      ;;
    *)
      SCRIPT_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${SCRIPT_ARGS[@]}"

if [[ -n "${GPU_ARG}" ]]; then
  GPU_ID="${GPU_ARG}"
fi

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

DATASET_PATH="${DATASET_PATH:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/extracted/exit-model/data/prepared_dataset/calibration/dolly15k_llama31.jsonl}"
METADATA="${METADATA:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/extracted/exit-model/artifacts/trained-exit-heads_kd_round_final/metadata.json}"
THRESHOLD_BASE_DIR="${THRESHOLD_BASE_DIR:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/extracted/exit-model/artifacts/trained-exit-heads_kd_round_final}"

LIMIT_OPT=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_OPT=(--limit "${LIMIT}")
fi

VERBOSE_OPT=(--verbose)
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50}"

shopt -s nullglob

declare -a threshold_dirs=()
if (( ${#THRESHOLD_FILTER[@]} > 0 )); then
  for spec in "${THRESHOLD_FILTER[@]}"; do
    case "${spec}" in
      tr_0.*) candidate="${THRESHOLD_BASE_DIR}/${spec}" ;;
      *) candidate="${THRESHOLD_BASE_DIR}/tr_${spec}" ;;
    esac
    if [[ ! -d "${candidate}" ]]; then
      echo "[error] Threshold directory not found: ${candidate}" >&2
      exit 1
    fi
    threshold_dirs+=("${candidate}")
  done
else
  threshold_dirs=("${THRESHOLD_BASE_DIR}"/tr_0.*)
fi

if (( ${#threshold_dirs[@]} == 0 )); then
  echo "[error] No threshold directories found under ${THRESHOLD_BASE_DIR}" >&2
  exit 1
fi

for threshold_dir in "${threshold_dirs[@]}"; do
  [[ -d "${threshold_dir}" ]] || continue
  dir_name="$(basename "${threshold_dir}")"
  threshold_files=("${threshold_dir}"/*.json)
  if (( ${#threshold_files[@]} == 0 )); then
    echo "[warn] Skipping ${dir_name} (no *.json threshold files found)"
    continue
  fi

  for threshold_file in "${threshold_files[@]}"; do
    threshold_name="$(basename "${threshold_file}")"
    log_file="${LOG_DIR}/${dir_name}_${threshold_name}.log"
    echo "[log] Writing to ${log_file}"
    {
      echo "[runbook] HF_HOME      : ${HF_HOME}"
      echo "[runbook] Dataset path : ${DATASET_PATH}"
      echo "[runbook] Metadata     : ${METADATA}"
      echo "[runbook] Thresholds   : ${threshold_file}"
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
        --thresholds "${threshold_file}" \
        --dataset "${DATASET_PATH}" \
        --device "${DEVICE}" \
        --progress-interval "${PROGRESS_INTERVAL}" \
        "${VERBOSE_OPT[@]}" \
        "${LIMIT_OPT[@]}" \
        "$@"
    } |& tee "${log_file}"
  done
done
