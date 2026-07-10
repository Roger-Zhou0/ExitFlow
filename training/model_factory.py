"""Helpers for instantiating backbone and exit heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, GPT2Config

from training.kd_trainer import ExitHeadSpec


@dataclass
class ModelBundle:
    backbone: nn.Module
    exit_specs: list[ExitHeadSpec]
    input_device: torch.device


class ExitHeadModule(nn.Module):
    """Two-layer MLP head that maps hidden states to vocabulary logits."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        reference_head: nn.Module | None = None,
        inner_size: int | None = None,
    ) -> None:
        super().__init__()
        if inner_size is None:
            inner_size = hidden_size
        self.mlp = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        if reference_head is not None and hasattr(reference_head, "weight"):
            self.output.weight.data.copy_(reference_head.weight.detach())

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.output(self.mlp(hidden_state)) 


def load_model_bundle(
    model_name: str,
    *,
    device: str,
    exit_points: Iterable[Tuple[str, int]],
    device_map: str | dict[str, str] | None = None,
    torch_dtype: torch.dtype | None = None,
    shard_exits: bool = False,
) -> ModelBundle:
    """Load a causal LM backbone and attach randomly initialized exit heads."""
    if model_name == "debug/tiny-gpt2":
        config = GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_positions=64,
            vocab_size=64,
        )
        backbone = AutoModelForCausalLM.from_config(config)
    else:
        load_kwargs: dict[str, object] = {}
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        backbone = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device_map is None:
        if torch_dtype is not None:
            backbone.to(device=device, dtype=torch_dtype)
        else:
            backbone.to(device)
    backbone.eval()
    try:
        backbone_params = sum(p.numel() for p in backbone.parameters())
        print(
            f"[model_factory] Backbone parameters: {backbone_params:,} "
            f"(≈{backbone_params / 1e6:.2f}M) dtype={next(backbone.parameters()).dtype} "
            f"device_map={'sharded' if device_map else device}",
            flush=True,
        )
    except StopIteration:
        print("[model_factory] Backbone parameters: 0", flush=True)

    hidden_size = backbone.config.hidden_size
    vocab_size = backbone.config.vocab_size
    intermediate_size = getattr(backbone.config, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = getattr(backbone.config, "n_inner", None)
    if intermediate_size is None:
        intermediate_size = hidden_size
    exit_specs: list[ExitHeadSpec] = []
    reference_head = getattr(backbone, "lm_head", None)
    hf_device_map = getattr(backbone, "hf_device_map", None)

    def _coerce_device(identifier: object, fallback: str) -> torch.device:
        if identifier is None:
            return torch.device(fallback)
        if isinstance(identifier, torch.device):
            return identifier
        if isinstance(identifier, int):
            return torch.device(f"cuda:{identifier}")
        if isinstance(identifier, str):
            return torch.device(identifier)
        return torch.device(fallback)

    primary_device = torch.device(device)
    if isinstance(hf_device_map, dict) and hf_device_map:
        if "model.embed_tokens" in hf_device_map:
            primary_device = _coerce_device(hf_device_map["model.embed_tokens"], device)
        else:
            first_value = next(iter(hf_device_map.values()))
            primary_device = _coerce_device(first_value, device)

    def _exit_device(layer_index: int) -> torch.device:
        if not shard_exits or hf_device_map is None:
            return primary_device
        if isinstance(hf_device_map, dict):
            candidate_keys = [
                f"model.layers.{max(layer_index - 1, 0)}",
                f"model.layers.{layer_index}",
            ]
            for key in candidate_keys:
                if key in hf_device_map:
                    return _coerce_device(hf_device_map[key], device)
            first_layer_device = hf_device_map.get("model.layers.0")
            if first_layer_device is not None:
                return _coerce_device(first_layer_device, device)
        return primary_device

    for identifier, layer_index in exit_points:
        head = ExitHeadModule(
            hidden_size,
            vocab_size,
            reference_head=reference_head,
            inner_size=intermediate_size,
        )
        target_device = _exit_device(layer_index)
        if torch_dtype is not None:
            head.to(device=target_device, dtype=torch_dtype)
        else:
            head.to(device=target_device)
        head_params = sum(p.numel() for p in head.parameters())
        print(
            f"[model_factory] Exit head {identifier} "
            f"params: {head_params:,} (≈{head_params / 1e6:.2f}M) device={target_device}",
            flush=True,
        )
        exit_specs.append(
            ExitHeadSpec(
                identifier=identifier,
                layer_index=layer_index,
                hidden_size=hidden_size,
                head=head,
            )
        )
    return ModelBundle(backbone=backbone, exit_specs=exit_specs, input_device=primary_device)
