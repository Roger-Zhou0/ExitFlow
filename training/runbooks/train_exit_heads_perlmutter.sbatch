#!/bin/bash
#
# Example SLURM batch script for training exit heads on Perlmutter.
# Submit with:  sbatch training/runbooks/train_exit_heads_perlmutter.sbatch
#
# Replace <account> and adjust time/queue as needed.

#SBATCH -J exit-head-train
#SBATCH -A m4431_g
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train_exit_heads_%j.out
#SBATCH --error=logs/train_exit_heads_%j.err

set -euo pipefail

module load conda
conda activate ee

mkdir -p logs

# Point to the aggregated training/calibration directories by default.
export DATASET_PATH=${DATASET_PATH:-/pscratch/sd/${USER:0:1}/${USER}/exit-model/data/prepared_dataset/train}
export CALIB_PATH=${CALIB_PATH:-/pscratch/sd/${USER:0:1}/${USER}/exit-model/data/prepared_dataset/calibration}

# Default to sharded optimizer state (matches recommended interactive run settings).
export OPTIMIZER_SHARDING=${OPTIMIZER_SHARDING:-sharded}
# Avoid gathering gigantic optimizer states in checkpoints by default.
# export SAVE_OPT_STATE=${SAVE_OPT_STATE:-0}
# Mitigate allocator fragmentation on long runs.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Ensure torchrun picks up all requested GPUs.
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# Optional overrides:
# export OUTPUT_DIR=...
# export CHECKPOINT_DIR=...
export SAVE_OPT_STATE=1      # to force full optimizer state snapshots
# export OPTIMIZER_SHARDING=none
# export PYTORCH_CUDA_ALLOC_CONF=...

bash training/runbooks/train_exit_heads_perlmutter.sh "$@"
