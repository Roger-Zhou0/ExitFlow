"""Checkpoint utilities for long-running KD training jobs."""

from __future__ import annotations

import random
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, Optional

import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from training.kd_trainer import ExitHeadSpec


@dataclass
class CheckpointState:
    epoch: int
    sample_index: int
    global_step: int
    accumulation_offset: int


class CheckpointManager:
    """Handles saving and restoring checkpoints with retention policy."""

    def __init__(self, directory: Path, keep: int = 5) -> None:
        self.directory = directory
        self.keep = keep
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {key: CheckpointManager._to_cpu(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [CheckpointManager._to_cpu(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(CheckpointManager._to_cpu(value) for value in obj)
        return obj

    @staticmethod
    def _is_meta_file(path: Path) -> bool:
        return bool(re.fullmatch(r"step_\d{8}\.pt", path.name))

    @staticmethod
    def _step_identifier(path: Path) -> Optional[str]:
        match = re.fullmatch(r"step_(\d{8})(?:_optim_rank\d{2})?\.pt", path.name)
        return match.group(1) if match else None

    def _list_meta_files(self) -> list[Path]:
        return sorted(p for p in self.directory.glob("step_*.pt") if self._is_meta_file(p))

    def latest_path(self) -> Optional[Path]:
        checkpoints = self._list_meta_files()
        return checkpoints[-1] if checkpoints else None

    def optimizer_shard_path(self, global_step: int, rank: int) -> Path:
        """Return the path to the sharded optimizer state for a given step/rank."""
        return self.directory / f"step_{global_step:08d}_optim_rank{rank:02d}.pt"

    def save_optimizer_shard(self, *, global_step: int, rank: int, state: dict) -> Path:
        """Persist a sharded optimizer state produced by ZeroRedundancyOptimizer."""
        path = self.optimizer_shard_path(global_step, rank)
        torch.save(self._to_cpu(state), path)
        return path

    def save(
        self,
        *,
        epoch: int,
        sample_index: int,
        global_step: int,
        accumulation_offset: int,
        backbone: torch.nn.Module,
        exits: list["ExitHeadSpec"],
        optimizer: torch.optim.Optimizer,
        optimizer_state: dict | None = None,
        save_optimizer: bool = True,
        optimizer_sharded: bool = False,
    ) -> Path:
        device_type = "cpu"
        for param in backbone.parameters():
            device_type = param.device.type
            break
        if save_optimizer:
            if optimizer_state is None:
                optimizer_state = optimizer.state_dict()

        state: Dict[str, object] = {
            "epoch": epoch,
            "sample_index": sample_index,
            "global_step": global_step,
            "accumulation_offset": accumulation_offset,
            "exit_heads": {spec.identifier: spec.head.state_dict() for spec in exits},
            "random_state": random.getstate(),
            "torch_state": torch.random.get_rng_state(),
            "device": device_type,
        }
        if optimizer_sharded:
            state["optimizer_sharded"] = True
        elif save_optimizer and optimizer_state is not None:
            state["optimizer"] = optimizer_state
        if torch.cuda.is_available():
            state["cuda_state"] = torch.cuda.get_rng_state_all()
        if hasattr(backbone, "state_dict"):
            state["backbone"] = backbone.state_dict()
        path = self.directory / f"step_{global_step:08d}.pt"
        torch.save(state, path)
        self._prune()
        return path

    def restore(
        self,
        *,
        backbone: torch.nn.Module,
        exits: list["ExitHeadSpec"],
        optimizer: torch.optim.Optimizer,
        rank: int | None = None,
    ) -> Optional[CheckpointState]:
        checkpoint_path = self.latest_path()
        if checkpoint_path is None:
            return None
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "backbone" in state:
            backbone.load_state_dict(state["backbone"])
        for spec in exits:
            head_state = state["exit_heads"].get(spec.identifier)
            if head_state is not None:
                spec.head.load_state_dict(head_state)
        optimizer_sharded = state.get("optimizer_sharded", False)
        if optimizer_sharded:
            if rank is None:
                raise ValueError("Rank must be provided to restore sharded optimizer state")
            shard_path = self.optimizer_shard_path(state["global_step"], rank)
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing optimizer shard for rank {rank}: {shard_path}")
            shard_state = torch.load(shard_path, map_location="cpu")
            target_optimizer = getattr(optimizer, "optim", optimizer)
            target_optimizer.load_state_dict(shard_state)
        elif "optimizer" in state:
            try:
                optimizer.load_local_state_dict(state["optimizer"])  # type: ignore[attr-defined]
            except AttributeError:
                optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["random_state"])
        torch.random.set_rng_state(state["torch_state"])
        if torch.cuda.is_available() and "cuda_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_state"])
        return CheckpointState(
            epoch=state["epoch"],
            sample_index=state["sample_index"],
            global_step=state["global_step"],
            accumulation_offset=state.get("accumulation_offset", 0),
        )

    def _prune(self) -> None:
        meta_files = self._list_meta_files()
        excess = len(meta_files) - self.keep
        if excess <= 0:
            return
        for path in meta_files[:excess]:
            step_id = self._step_identifier(path)
            path.unlink(missing_ok=True)
            if step_id is None:
                continue
            for shard_file in self.directory.glob(f"step_{step_id}_optim_rank*.pt"):
                shard_file.unlink(missing_ok=True)
