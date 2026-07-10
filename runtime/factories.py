"""Factory helpers for building segments and exit heads from weights."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import create_causal_mask

from .segment_worker import SegmentModule


def _load_torchscript_module(path: Path, device: str) -> torch.jit.ScriptModule:
    if not path.exists():
        raise FileNotFoundError(f"Missing weights file: {path}")
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    return module


def torch_segment_factory(options: Mapping[str, Any]) -> SegmentModule:
    """Instantiate a segment module from TorchScript weights."""
    artifacts = options.get("artifacts", {})
    weights_path = artifacts.get("segment_weights")
    if weights_path is None:
        raise ValueError(f"Segment {options.get('segment_id')} missing segment_weights artifact")
    module = _load_torchscript_module(Path(weights_path), options["device"])
    return module


def torch_exit_head_factory(options: Mapping[str, Any]) -> torch.nn.Module:
    """Instantiate an exit head module from TorchScript weights."""
    weights_path = options.get("exit_head_weights")
    if weights_path is None:
        raise ValueError(f"Forwarder {options.get('forwarder_id')} missing exit_head_weights artifact")
    module = _load_torchscript_module(Path(weights_path), options["device"])
    return module


_BACKBONE_CACHE: dict[tuple[str, str, str], AutoModelForCausalLM] = {}


def _get_backbone(backbone_dir: str, device: str, dtype: str) -> AutoModelForCausalLM:
    """Load and cache the HF backbone per (path, device, dtype)."""
    key = (str(Path(backbone_dir).resolve()), device, dtype)
    if key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[key]
    torch_dtype = torch.float32 if dtype in {"fp32", "float32"} else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        backbone_dir,
        torch_dtype=torch_dtype,
        device_map=None,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    _BACKBONE_CACHE[key] = model
    return model


class HFSegment(torch.nn.Module):
    """Native HF segment sliced from the full backbone."""

    def __init__(
        self,
        segment_id: str,
        backbone: AutoModelForCausalLM,
        start_idx: int,
        end_idx: int,
        include_embed: bool,
        include_norm: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.seg_id = segment_id
        self.device = torch.device(device)
        layers = backbone.model.layers
        self.layers = torch.nn.ModuleList(layers[start_idx:end_idx]).to(self.device)
        self.embed = backbone.model.embed_tokens.to(self.device) if include_embed else None
        self.norm = backbone.model.norm.to(self.device) if include_norm else None
        self.rotary = backbone.model.rotary_emb.to(self.device)
        self.config = backbone.config
        self.debug = os.getenv("SEGMENT_DEBUG", "0") == "1"
        self._logged = False
        first_param = next(self.layers[0].parameters(), None)
        self.target_dtype = first_param.dtype if first_param is not None else torch.float32

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Accept either token IDs (int) or hidden states (float).
        if hidden_state.dtype in (torch.int64, torch.int32):
            if self.embed is None:
                raise ValueError("Segment received token ids but embed layer is not available")
            input_ids = hidden_state.to(dtype=torch.long, device=self.device)
            hidden_state = self.embed(input_ids)
        else:
            hidden_state = hidden_state.to(self.device)

        hidden_state = hidden_state.to(dtype=self.target_dtype)

        batch, seq_len, _ = hidden_state.shape
        cache_position = torch.arange(seq_len, device=self.device)
        position_ids = cache_position.unsqueeze(0)
        # attention_mask: [bs, seq] -> bool
        attention_mask = torch.ones((batch, seq_len), device=self.device, dtype=torch.bool)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_state,
            attention_mask=attention_mask,
            past_key_values=None,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary(hidden_state, position_ids)
        if self.debug and not self._logged:
            print(
                f"[segment {self.seg_id}] device={self.device} dtype={hidden_state.dtype} "
                f"seq_len={seq_len} layers={len(self.layers)}"
            )
            self._logged = True
        for block in self.layers:
            hidden_state = block(
                hidden_states=hidden_state,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
            )
            # block may return tuple (hidden_states, attn_weights); take first element if tuple
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
        if self.norm is not None:
            hidden_state = self.norm(hidden_state)
        return hidden_state


def hf_segment_factory(options: Mapping[str, Any]) -> SegmentModule:
    """Instantiate a native HF segment by slicing the backbone layers."""
    backbone_dir = options.get("backbone_dir")
    if backbone_dir is None:
        raise ValueError("native segment requires backbone_dir")
    device = options.get("device", "cpu")
    dtype = options.get("dtype", "float16")
    layers = options.get("layers", {})
    start = int(layers.get("start", 1)) - 1  # convert to 0-based
    end = int(layers.get("end", start + 1))
    include_embed = options.get("include_embed", start == 0)
    include_norm = options.get("include_norm", False)
    backbone = _get_backbone(backbone_dir, device, dtype)
    return HFSegment(options.get("segment_id", "segment"), backbone, start, end, include_embed, include_norm, device)


@dataclass
class RuntimeFactories:
    """Convenience bundle for standard Torch-based factories."""

    segment_factory: Callable[[Mapping[str, Any]], SegmentModule] = torch_segment_factory
    exit_head_factory: Callable[[Mapping[str, Any]], torch.nn.Module] = torch_exit_head_factory
    native_segment_factory: Callable[[Mapping[str, Any]], SegmentModule] = hf_segment_factory


__all__ = [
    "RuntimeFactories",
    "torch_exit_head_factory",
    "torch_segment_factory",
    "hf_segment_factory",
]
