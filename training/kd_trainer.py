"""Knowledge distillation training loop for exit heads."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Iterable, Protocol

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from training.checkpoint import CheckpointState


class BackboneModel(Protocol):
    """Protocol for the frozen transformer backbone."""

    def forward(self, input_ids: torch.Tensor, *, output_hidden_states: bool = False) -> BackboneOutput:
        ...

    def parameters(self):
        ...

    def eval(self) -> None:
        ...


class ExitHead(Protocol):
    """Protocol describing an attached exit head module."""

    def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...

    def parameters(self):
        ...

    def train(self) -> None:
        ...


class BackboneOutput(Protocol):
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


@dataclass
class ExitHeadSpec:
    """Metadata for locating the hidden state feeding an exit head."""

    identifier: str
    layer_index: int
    hidden_size: int
    head: ExitHead


@dataclass
class KDConfig:
    temperature: float
    exit_loss_weights: dict[str, float]
    ce_loss_weight: float = 1.0
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None
    epochs: int = 1
    frozen_backbone: bool = True
    optimizer: str = "adamw"
    device: str = "cuda"
    # Optional DeepSpeed integration
    deepspeed: bool = False
    deepspeed_config_path: "str | None" = None


class KnowledgeDistillationTrainer:
    """Runs KD for exit heads while keeping the backbone frozen."""

    def __init__(
        self,
        backbone: BackboneModel,
        exits: list[ExitHeadSpec],
        config: KDConfig,
    ) -> None:
        self.backbone = backbone
        self.exits = exits
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self._using_zero = False
        self.ds_engine = None

    def train(
        self,
        samples: Iterable[dict[str, list[int]]],
        *,
        checkpoint_manager: "CheckpointManager | None" = None,
        checkpoint_dir: "Path | None" = None,
        checkpoint_interval: int = 100,
        resume: bool = False,
        shuffle: bool = True,
        epochs_override: int | None = None,
        max_steps_override: int | None = None,
    ) -> None:
        """Iterate over the dataset and update exit heads."""
        samples_iterable = samples
        self._prepare_backbone()
        optimizer = self._build_optimizer()

        # Optional DeepSpeed initialisation (wrap exit heads into a container module)
        if self.config.deepspeed:
            try:
                import deepspeed  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dep
                raise RuntimeError("DeepSpeed requested but not available (pip install deepspeed)") from exc

            class ExitHeadsContainer(torch.nn.Module):
                def __init__(self, exits: list[ExitHeadSpec]):
                    super().__init__()
                    self.heads = torch.nn.ModuleDict({spec.identifier: spec.head for spec in exits})

                def forward(self, *args, **kwargs):  # not used; DS handles backward on external loss
                    return None

            container = ExitHeadsContainer(self.exits)
            ds_config = None
            if self.config.deepspeed_config_path:
                with open(self.config.deepspeed_config_path, "r", encoding="utf-8") as fh:
                    ds_config = json.load(fh)
            if ds_config is None:
                ds_config = {
                    "bf16": {"enabled": True},
                    "zero_optimization": {"stage": 3},
                    "optimizer": {"type": "AdamW", "params": {"lr": float(self.config.learning_rate), "torch_adam": True}}
                }
            # Ensure optimizer is provided via DS config to avoid client optimizer with offload
            if "optimizer" not in ds_config:
                ds_config["optimizer"] = {"type": "AdamW", "params": {"lr": float(self.config.learning_rate), "torch_adam": True}}
            else:
                # Keep DS config as source of truth but align LR if not set
                ds_params = ds_config.setdefault("optimizer", {}).setdefault("params", {})
                ds_params.setdefault("lr", float(self.config.learning_rate))
                ds_params.setdefault("torch_adam", True)
            self.ds_engine, optimizer, _, _ = deepspeed.initialize(
                model=container,
                model_parameters=list(container.parameters()),
                config=ds_config,
            )
            self._using_zero = True
        self._using_zero = isinstance(optimizer, ZeroRedundancyOptimizer)
        resume_state = None
        if resume:
            if self.config.deepspeed and self.ds_engine and checkpoint_dir is not None:
                # Let DeepSpeed restore model/optimizer shards and read back client_state
                result = self.ds_engine.load_checkpoint(str(checkpoint_dir))
                client_state = {}
                if isinstance(result, tuple) and len(result) >= 2:
                    _, client_state = result[0], result[1]
                elif isinstance(result, dict):  # some versions may return dict
                    client_state = result
                if client_state:
                    resume_state = CheckpointState(
                        epoch=client_state.get("epoch", 0),
                        sample_index=client_state.get("sample_index", 0),
                        global_step=client_state.get("global_step", 0),
                        accumulation_offset=client_state.get("accumulation_offset", 0),
                    )
            elif checkpoint_manager:
                resume_state = checkpoint_manager.restore(
                    backbone=self.backbone,
                    exits=self.exits,
                    optimizer=optimizer,
                    rank=self.rank,
                )
        global_step = resume_state.global_step if resume_state else 0
        accumulation_offset = resume_state.accumulation_offset if resume_state else 0
        start_epoch = resume_state.epoch if resume_state else 0
        sample_offset = resume_state.sample_index if resume_state else 0
        epochs = epochs_override or self.config.epochs
        max_steps_limit = max_steps_override or self.config.max_steps

        samples_list: list[dict[str, list[int]]] | None = None
        if shuffle:
            samples_list = list(samples_iterable)
            if not samples_list:
                return

        for epoch in range(start_epoch, epochs):
            for spec in self.exits:
                spec.head.train()

            if shuffle and samples_list is not None:
                indices = list(range(len(samples_list)))
                random.shuffle(indices)
                batch_iter = ((position, samples_list[index]) for position, index in enumerate(indices))
            else:
                iterator = iter(samples_iterable)
                if epoch == start_epoch and sample_offset:
                    iterator = itertools.islice(iterator, sample_offset, None)
                    start_position = sample_offset
                else:
                    start_position = 0
                batch_iter = enumerate(iterator, start=start_position)

            processed_any = False
            for position, batch in batch_iter:
                if shuffle and epoch == start_epoch and position < sample_offset:
                    continue
                processed_any = True
                loss = self._forward_batch(batch)
                if self.ds_engine is not None:
                    self.ds_engine.backward(loss)
                else:
                    loss.backward()
                accumulation_offset += 1
                if self.rank == 0 and global_step % 10 == 0:
                    print(
                        f"[trainer][rank {self.rank}] epoch {epoch} position {position} loss {loss.item():.4f}",
                        flush=True,
                    )
                if accumulation_offset % self.config.gradient_accumulation_steps == 0:
                    if self.ds_engine is not None:
                        self.ds_engine.step()
                        # Some DeepSpeed versions do not accept set_to_none argument
                        self.ds_engine.zero_grad()
                    else:
                        self._allreduce_gradients()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accumulation_offset = 0
                    # Synchronized checkpoint: ensure all ranks participate when any rank reaches interval
                    if checkpoint_manager or (self.ds_engine is not None and checkpoint_dir is not None):
                        local_trigger = 1 if (global_step % checkpoint_interval == 0) else 0
                        should_checkpoint = local_trigger == 1
                        if self.distributed:
                            flag = torch.tensor([local_trigger], device=self.device)
                            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                            should_checkpoint = bool(flag.item() > 0)
                        if should_checkpoint:
                            if self.ds_engine is not None and checkpoint_dir is not None:
                                # Use DeepSpeed checkpointing (sharded, offload-aware)
                                client_state = {
                                    "epoch": epoch,
                                    "sample_index": position,
                                    "global_step": global_step,
                                    "accumulation_offset": accumulation_offset,
                                }
                                # Only rank0 will print; DS manages collective save
                                self.ds_engine.save_checkpoint(
                                    str(checkpoint_dir),
                                    tag=f"global_step_{global_step:08d}",
                                    client_state=client_state,
                                )
                                if self.distributed:
                                    dist.barrier()
                            elif checkpoint_manager is not None:
                                save_opt = os.environ.get("SAVE_OPT_STATE", "1").lower() not in {"0", "false", "no"}
                                optimizer_state = None
                                optimizer_sharded = False
                                if save_opt:
                                    if self._using_zero:
                                        optimizer_sharded = True
                                        shard_state = optimizer.optim.state_dict()  # type: ignore[attr-defined]
                                        checkpoint_manager.save_optimizer_shard(
                                            global_step=global_step,
                                            rank=self.rank,
                                            state=shard_state,
                                        )
                                    elif self.rank == 0:
                                        optimizer_state = optimizer.state_dict()
                                if self.rank == 0:
                                    checkpoint_manager.save(
                                        epoch=epoch,
                                        sample_index=position,
                                        global_step=global_step,
                                        accumulation_offset=accumulation_offset,
                                        backbone=self.backbone,
                                        exits=self.exits,
                                        optimizer=optimizer,
                                        optimizer_state=optimizer_state,
                                        save_optimizer=save_opt and not optimizer_sharded,
                                        optimizer_sharded=optimizer_sharded,
                                    )
                                if self.distributed:
                                    dist.barrier()
                    if max_steps_limit and global_step >= max_steps_limit:
                        return
            if not processed_any:
                if self.rank == 0:
                    print(f"[trainer][rank {self.rank}] No batches available; terminating early.", flush=True)
                break
            sample_offset = 0

    def _prepare_backbone(self) -> None:
        """Freeze backbone parameters if requested."""
        if hasattr(self.backbone, "eval"):
            self.backbone.eval()
        if not self.config.frozen_backbone:
            return
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer for exit heads."""
        params: list[torch.nn.Parameter] = []
        for spec in self.exits:
            params.extend(list(spec.head.parameters()))
        if not params:
            raise ValueError("No exit head parameters provided")
        optimizer_name = self.config.optimizer.lower()
        # In distributed mode, prefer simple data-parallel with manual all-reduce unless explicitly
        # requested to use ZeRO via OPTIMIZER_SHARDING env var. Mixing ZeRO with manual all-reduce
        # can cause collective mismatches/timeouts.
        if self.distributed:
            shard_mode = os.environ.get("OPTIMIZER_SHARDING", "none").lower()
            if shard_mode in {"sharded", "zero", "zerodp"}:
                optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
                return ZeroRedundancyOptimizer(
                    params,
                    optimizer_class=optimizer_class,
                    lr=self.config.learning_rate,
                    overlap_with_ddp=False,
                )
            # Fall back to regular optimizer per-rank; gradients are averaged manually.
            if optimizer_name == "adamw":
                return torch.optim.AdamW(params, lr=self.config.learning_rate)
            if optimizer_name == "adam":
                return torch.optim.Adam(params, lr=self.config.learning_rate)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=self.config.learning_rate)
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self.config.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _forward_batch(self, batch: dict[str, list[int]]) -> torch.Tensor:
        """Compute loss for all exit heads using KD (+ optional CE)."""
        input_sequences = batch["input_ids"]
        label_sequences = batch["labels"]
        if input_sequences and isinstance(input_sequences[0], int):
            input_sequences = [input_sequences]
            label_sequences = [label_sequences]

        input_tensors = [torch.tensor(sample, device=self.device, dtype=torch.long) for sample in input_sequences]
        label_tensors = [torch.tensor(sample, device=self.device, dtype=torch.long) for sample in label_sequences]
        lengths = torch.tensor([tensor.size(0) for tensor in input_tensors], device=self.device)
        if lengths.numel() == 0:
            raise ValueError("Received empty batch")

        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=0)
        labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)

        outputs = self.backbone.forward(input_ids, output_hidden_states=True)
        final_logits = outputs.logits
        hidden_states = outputs.hidden_states

        teacher_device = final_logits.device
        # Teacher distribution for all time steps except the last (shifted language modeling)
        if final_logits.dim() == 3 and final_logits.size(1) > 1:
            teacher_logits_seq = final_logits[:, :-1, :]
        else:
            # No valid steps to supervise (seq_len <= 1)
            teacher_logits_seq = final_logits[:, :0, :]
        teacher_distribution = F.softmax(teacher_logits_seq.detach() / self.config.temperature, dim=-1)
        # Ensure labels live on the same device for CE indexing if used
        labels_on_teacher = labels.to(teacher_device)

        total_loss = torch.zeros((), device=teacher_device)
        for spec in self.exits:
            hidden_state = hidden_states[spec.layer_index]
            target_device = hidden_state.device
            if hasattr(spec.head, "parameters"):
                try:
                    current_param = next(spec.head.parameters())
                except StopIteration:
                    current_param = None
                if current_param is not None:
                    if current_param.device != target_device or current_param.dtype != hidden_state.dtype:
                        spec.head.to(device=target_device, dtype=hidden_state.dtype)
            # Use full-sequence hidden states aligned to teacher: positions [0..T-2]
            if hidden_state.dim() == 2:
                # Shape [B, H] -> treat as sequence length 1; after slicing becomes 0 steps
                hidden_state = hidden_state.unsqueeze(1)
            if hidden_state.size(1) > 1:
                hidden_state_seq = hidden_state[:, :-1, :]
            else:
                hidden_state_seq = hidden_state[:, :0, :]
            student_logits = spec.head(hidden_state_seq).to(teacher_device)

            # KD loss (temperature-scaled KL) over sequence with mask
            student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=-1)
            # Per-position KL: [B, T-1, V] -> sum over vocab -> [B, T-1]
            kd_elementwise = F.kl_div(
                student_log_probs,
                teacher_distribution,
                reduction="none",
            )
            kd_per_pos = kd_elementwise.sum(dim=-1)

            # Valid supervision mask comes from shifted labels (labels[:, 1:])
            if labels_on_teacher.size(1) > 1:
                targets_seq = labels_on_teacher[:, 1:]
            else:
                targets_seq = labels_on_teacher[:, :0]
            valid_mask = (targets_seq != -100).to(dtype=kd_per_pos.dtype)
            valid_count = valid_mask.sum().clamp(min=1.0)
            kd_loss = (kd_per_pos * valid_mask).sum() / valid_count

            # Optional CE loss over sequence (ignore_index=-100); mean over valid tokens
            ce_loss = torch.zeros_like(kd_loss)
            if self.config.ce_loss_weight != 0.0:
                if student_logits.numel() == 0:
                    ce_loss = torch.zeros((), device=teacher_device)
                else:
                    B, Tm1, V = student_logits.shape
                    ce_loss = F.cross_entropy(
                        student_logits.reshape(B * Tm1, V),
                        targets_seq.reshape(B * Tm1),
                        ignore_index=-100,
                        reduction="mean",
                    )

            weight = self.config.exit_loss_weights.get(spec.identifier, 1.0)
            # Scale KD by T^2 only; mix in CE via ce_loss_weight
            total_loss = total_loss + weight * (
                kd_loss * (self.config.temperature ** 2) + self.config.ce_loss_weight * ce_loss
            )

        return total_loss

    def _allreduce_gradients(self) -> None:
        if not self.distributed or self.ds_engine is not None:
            return
        for spec in self.exits:
            for param in spec.head.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)
