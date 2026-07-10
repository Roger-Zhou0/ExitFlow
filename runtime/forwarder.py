"""Forwarder logic for early exit decisions and routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class ExitHead(Protocol):
    """Protocol for computing logits at an exit head."""

    def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class ForwarderDecision:
    """Result of evaluating one early-exit head."""

    exit_now: bool
    next_segment_id: str | None
    forwarded_state: torch.Tensor | None
    logits: torch.Tensor | None
    scores: dict[str, float]


@dataclass(frozen=True)
class ForwarderConfig:
    """Calibrated gates and routing target for an exit head."""

    logit_threshold: float
    margin_gate: float
    entropy_gate: float
    temperature_scale: float
    next_segment_id: str | None


class Forwarder:
    """Evaluates exit head logits and decides whether to exit or forward."""

    def __init__(
        self,
        exit_head: ExitHead,
        config: ForwarderConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.exit_head = exit_head
        self.config = config
        self.device = torch.device(device) if device is not None else None

        if config.temperature_scale <= 0:
            raise ValueError("temperature_scale must be greater than zero")

        # Keep the exit head on the same device as its segment when requested.
        if hasattr(self.exit_head, "to") and self.device is not None:
            self.exit_head = self.exit_head.to(self.device)  # type: ignore[assignment]
        if hasattr(self.exit_head, "eval"):
            self.exit_head.eval()  # type: ignore[call-arg]

    @torch.inference_mode()
    def evaluate(self, hidden_state: torch.Tensor) -> ForwarderDecision:
        """Run the exit head and compute the decision gates."""
        x = hidden_state

        # Move activations to the exit head without blocking when possible.
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)

        x = self._coerce_hidden_for_head(x)
        logits_full = self.exit_head(x)
        token_logits = self._select_token_logits(logits_full)

        # Gate on CPU so the forwarder does not retain a vocabulary-sized GPU
        # tensor after making its decision.
        if token_logits.is_cuda:
            token_logits_cpu = token_logits.float().cpu()
        else:
            token_logits_cpu = token_logits.float()

        scaled = token_logits_cpu / self.config.temperature_scale
        probs = F.softmax(scaled, dim=-1)

        topk = torch.topk(scaled, k=min(2, scaled.shape[-1]))
        top_logit = float(topk.values[0].item())
        margin = float((topk.values[0] - topk.values[1]).item()) if topk.values.numel() > 1 else float("inf")
        entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())

        exit_now = (
            top_logit >= self.config.logit_threshold
            and margin >= self.config.margin_gate
            and entropy <= self.config.entropy_gate
        )

        next_segment_id = None if exit_now else self.config.next_segment_id

        forwarded_state: torch.Tensor | None = None
        if not exit_now:
            forwarded_state = x

        scores = {
            "logit": top_logit,
            "margin": margin,
            "entropy": entropy,
        }

        logits_out: torch.Tensor | None = token_logits_cpu if exit_now else None

        return ForwarderDecision(
            exit_now=exit_now,
            next_segment_id=next_segment_id,
            forwarded_state=forwarded_state,
            logits=logits_out,
            scores=scores,
        )

    @staticmethod
    def _select_token_logits(logits: torch.Tensor) -> torch.Tensor:
        """Pick logits for the last token as a 1D tensor."""
        if logits.dim() == 3:
            token_logits = logits[:, -1, :]
        else:
            token_logits = logits
        if token_logits.dim() == 2 and token_logits.shape[0] == 1:
            token_logits = token_logits.squeeze(0)
        return token_logits

    def _coerce_hidden_for_head(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Best-effort reshape/pad to match exit head expected hidden size."""
        target_param = next(self.exit_head.parameters(), None) if hasattr(self.exit_head, "parameters") else None
        if target_param is None:
            return hidden_state

        target_device = target_param.device
        target_dtype = target_param.dtype

        # Assume Linear-like shape [vocab, hidden_size] or [hidden_size, vocab]
        if target_param.dim() >= 2:
            hidden_size = target_param.shape[1] if target_param.shape[1] > 0 else target_param.shape[0]
        else:
            hidden_size = target_param.numel()

        tensor = hidden_state.to(device=target_device, dtype=target_dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        last_dim = tensor.shape[-1]
        if last_dim == hidden_size:
            return tensor
        if last_dim > hidden_size:
            return tensor[..., :hidden_size]

        # pad to match expected hidden size
        pad_width = hidden_size - last_dim
        pad_shape = (0, pad_width)
        tensor = F.pad(tensor, pad_shape)
        return tensor
