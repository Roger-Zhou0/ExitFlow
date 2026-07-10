#!/bin/bash
#SBATCH --job-name=exit-head-kd
# On Perlmutter, prefer -C gpu and --gpus-per-task over --gres
#SBATCH -C gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# By default write logs to submission directory; avoid requiring pre-made folders
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module load cuda/12.1 || true
module load anaconda || true

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Activate environment containing PyTorch + transformers
source "$HOME/.bashrc"
conda activate llama-kd

# Defaults (can be overridden by CLI flags below)
MANIFEST="config/manifest.json"
DATASET="data/training/dataset.jsonl"
OUTPUT_DIR="artifacts/exit-heads"
# Default checkpoint directory points to trained-exit-heads_kd_round0
CHECKPOINT_DIR="checkpoints/trained-exit-heads_kd_round0"
CALIBRATION="data/training/calibration.jsonl"
THRESHOLDS="artifacts/exit-heads/thresholds.json"
DEVICE="cuda"
BATCH_SIZE=4
SAVE_INTERVAL=500
RESUME_FLAG=false
# Allow enabling resume via environment variable: export RESUME=1
if [[ -n "${RESUME:-}" && "${RESUME}" != "0" ]]; then
  RESUME_FLAG=true
fi
MAX_STEPS=""
EPOCHS=""

# Parse CLI flags to avoid relying on environment variables
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoints)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    --calibration-path)
      CALIBRATION="$2"; shift 2 ;;
    --thresholds-output)
      THRESHOLDS="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --save-interval)
      SAVE_INTERVAL="$2"; shift 2 ;;
    --max-steps)
      MAX_STEPS="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --resume)
      RESUME_FLAG=true; shift 1 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" logs

# Quick diagnostics to verify GPU allocation and CUDA visibility
echo "===== DIAGNOSTICS =====" >&2
hostname >&2 || true
nvidia-smi >&2 || echo "nvidia-smi not available" >&2
python - << 'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("visible devices:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "<none>")
PY
echo "=======================" >&2

CMD=(
  python -m training.main
  --manifest "$MANIFEST"
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --checkpoints "$CHECKPOINT_DIR"
  --calibration-path "$CALIBRATION"
  --thresholds-output "$THRESHOLDS"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --save-interval "$SAVE_INTERVAL"
)

if [[ -n "$MAX_STEPS" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$EPOCHS" ]]; then
  CMD+=(--epochs "$EPOCHS")
fi
if [[ "$RESUME_FLAG" == true ]]; then
  CMD+=(--resume)
fi

echo "Running: ${CMD[*]}" >&2
"${CMD[@]}"
