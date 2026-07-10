#!/bin/bash
#
# Example SLURM batch script for training exit heads on Perlmutter.
# Submit with:  sbatch training/runbooks/train_exit_heads_perlmutter.sbatch
#
# Replace <account> and adjust time/queue as needed.

#SBATCH -J exit-head-train
#SBATCH -A m4431_g
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train_exit_heads_%j.out
#SBATCH --error=logs/train_exit_heads_%j.err

set -euo pipefail

module load conda
# 你的环境名
conda activate ddl

mkdir -p logs

# → 路径改为你的 /extracted 前缀
export DATASET_PATH=${DATASET_PATH:-/pscratch/sd/${USER:0:1}/${USER}/extracted/exit-model/data/prepared_dataset/train}
export CALIB_PATH=${CALIB_PATH:-/pscratch/sd/${USER:0:1}/${USER}/extracted/exit-model/data/prepared_dataset/calibration}

# 与朋友一致：默认启用优化器状态分片
export OPTIMIZER_SHARDING=${OPTIMIZER_SHARDING:-sharded}
# 避免分布式聚合巨大优化器状态（他默认注释掉；你们现在要“完整保存优化器状态”，所以下面强制 1）
export SAVE_OPT_STATE=${SAVE_OPT_STATE:-1}

# 内存分配器碎片缓解
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# torchrun 进程数 / GPU 数
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# 可选覆盖：
# export OUTPUT_DIR=...
# export CHECKPOINT_DIR=...
# export OPTIMIZER_SHARDING=none
# export PYTORCH_CUDA_ALLOC_CONF=...

bash train_exit_heads_perlmutter.sh "$@"
