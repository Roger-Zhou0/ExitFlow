#!/usr/bin/env bash
#
# Launch exit-head training on Perlmutter with torchrun (4 GPUs by default).
# Assumes you have already obtained an interactive allocation via:
#   salloc -A <account> -q interactive -t 01:00:00 -C gpu -G 4
#
# Optional environment overrides:
#   ACCOUNT           - NERSC account (only used if --account flag appended manually)
#   HF_HOME           - Hugging Face cache directory (default: /pscratch/<user>/hf_cache)
#   DATASET_PATH      - Tokenised training dataset JSONL or directory
#   CALIB_PATH        - Calibration dataset JSONL or directory
#   OUTPUT_DIR        - Artifact output directory
#   CHECKPOINT_DIR    - Checkpoint directory
#   NPROC_PER_NODE    - Number of GPUs for torchrun (default: 4)
#   BATCH_SIZE        - Optimiser batch size (default: 1)
#   SAVE_INTERVAL     - Checkpoint interval in steps (default: 1000)
#   CHUNK_BYTES       - Optional streaming chunk cap (accepts plain integers or suffixes like 64G; default: 64G)
#   CHUNK_SAMPLES     - Optional streaming chunk sample cap (default: 32768)
#
# Additional CLI flags can be appended when invoking this script; they will be
# forwarded to training/distributed_main.py.

set -euo pipefail
module load conda

# Activate the environment that contains torch/transformers.
conda activate ee

USERNAME="${USER}"
readonly USERNAME

export HF_HOME="${HF_HOME:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_PATH="${DATASET_PATH:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/exit-model/data/prepared_dataset/train}"
CALIB_PATH="${CALIB_PATH:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/exit-model/data/prepared_dataset/calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/exit-model/artifacts/trained-exit-heads_kd_round_final}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/pscratch/sd/${USERNAME:0:1}/${USERNAME}/exit-model/checkpoints/trained-exit-heads_kd_round_final}"

NPROC="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
CHUNK_BYTES_RAW="${CHUNK_BYTES:-64G}"
CHUNK_SAMPLES_RAW="${CHUNK_SAMPLES:-32768}"

parse_size() {
  local value=$1
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value}"
    return
  fi
  if [[ "${value}" =~ ^([0-9]+)([KkMmGgTt])$ ]]; then
    local num=${BASH_REMATCH[1]}
    local unit=${BASH_REMATCH[2],,}
    case "${unit}" in
      k) echo $((num * 1024)) ;;
      m) echo $((num * 1024 * 1024)) ;;
      g) echo $((num * 1024 * 1024 * 1024)) ;;
      t) echo $((num * 1024 * 1024 * 1024 * 1024)) ;;
      *) return 1 ;;
    esac
    return
  fi
  return 1
}

CHUNK_ARGS=()
CHUNK_BYTES_VALUE=0
if [[ -n "${CHUNK_BYTES_RAW}" && "${CHUNK_BYTES_RAW}" != "0" ]]; then
  if ! CHUNK_BYTES_VALUE=$(parse_size "${CHUNK_BYTES_RAW}"); then
    echo "[runbook] Error: unable to parse CHUNK_BYTES='${CHUNK_BYTES_RAW}'" >&2
    exit 1
  fi
  CHUNK_ARGS+=(--dataset-chunk-bytes "${CHUNK_BYTES_VALUE}")
fi

if [[ -n "${CHUNK_SAMPLES_RAW}" && "${CHUNK_SAMPLES_RAW}" != "0" ]]; then
  if [[ ! "${CHUNK_SAMPLES_RAW}" =~ ^[0-9]+$ ]]; then
    echo "[runbook] Error: CHUNK_SAMPLES must be an integer (got '${CHUNK_SAMPLES_RAW}')" >&2
    exit 1
  fi
  CHUNK_ARGS+=(--dataset-chunk-samples "${CHUNK_SAMPLES_RAW}")
fi

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

echo "[runbook] HF_HOME           : ${HF_HOME}"
echo "[runbook] Dataset path      : ${DATASET_PATH}"
echo "[runbook] Calibration path  : ${CALIB_PATH}"
echo "[runbook] Output directory  : ${OUTPUT_DIR}"
echo "[runbook] Checkpoint dir    : ${CHECKPOINT_DIR}"
echo "[runbook] GPUs (nproc)      : ${NPROC}"
if [[ ${#CHUNK_ARGS[@]} -gt 0 ]]; then
  printf "[runbook] Streaming chunks  :"
  if (( CHUNK_BYTES_VALUE > 0 )); then
    printf " %s bytes" "${CHUNK_BYTES_VALUE}"
  fi
  if [[ -n "${CHUNK_SAMPLES_RAW}" && "${CHUNK_SAMPLES_RAW}" != "0" ]]; then
    printf " %s samples" "${CHUNK_SAMPLES_RAW}"
  fi
  printf "\n"
else
  echo "[runbook] Streaming chunks  : disabled"
fi

RESUME_ARGS=()
if [[ -n "${RESUME:-}" && "${RESUME}" != "0" ]]; then
  echo "[runbook] Resume            : enabled (--resume)"
  RESUME_ARGS+=(--resume)
else
  echo "[runbook] Resume            : disabled"
fi

TORCHRUN_ARGS=(
  --manifest config/manifest.json
  --dataset "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --checkpoints "${CHECKPOINT_DIR}"
  --calibration-path "${CALIB_PATH}"
  --thresholds-output "${OUTPUT_DIR}/thresholds.json"
  --batch-size "${BATCH_SIZE}"
  --save-interval "${SAVE_INTERVAL}"
)

if [[ -n "${DEEPSPEED:-}" && "${DEEPSPEED}" != "0" ]]; then
  TORCHRUN_ARGS+=(--deepspeed)
fi

if [[ -n "${DS_CONFIG:-}" ]]; then
  TORCHRUN_ARGS+=(--deepspeed-config "${DS_CONFIG}")
fi

if (( ${#CHUNK_ARGS[@]} )); then
  TORCHRUN_ARGS+=("${CHUNK_ARGS[@]}")
fi

if (( ${#RESUME_ARGS[@]} )); then
  TORCHRUN_ARGS+=("${RESUME_ARGS[@]}")
fi

if (( $# )); then
  TORCHRUN_ARGS+=("$@")
fi

torchrun --nproc_per_node="${NPROC}" training/distributed_main.py "${TORCHRUN_ARGS[@]}"
