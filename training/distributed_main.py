"""Distributed CLI entrypoint for exit-head KD training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import TrainingJobConfig
from training.job import run_training_job
from training.main import build_arg_parser


def _init_distributed(init_method: str) -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl", init_method=init_method)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    except RuntimeError:
        local_rank = rank
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument(
        "--init-method",
        default="env://",
        help="Torch.distributed init_method (default: env://)",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = _init_distributed(args.init_method)
    if rank == 0:
        print(f"[dist] World size: {world_size} | local rank: {local_rank}")

    config = TrainingJobConfig.from_args(args)
    config.device = f"cuda:{local_rank}"
    # Device maps conflict with per-rank sharding; disable for distributed runs.
    config.device_map = None
    config.shard_exits = True

    try:
        run_training_job(config)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
