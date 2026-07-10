"""Segment worker responsible for a subset of transformer layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .forwarder import Forwarder, ForwarderDecision


class SegmentModule(Protocol):
    """Protocol for the segment-specific transformer block."""

    def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class SegmentContext:
    """Identity and device placement for a segment worker."""

    segment_id: str
    device: str


class SegmentWorker:
    """Owns a transformer segment and its associated forwarder."""

    def __init__(
        self,
        context: SegmentContext,
        segment: SegmentModule,
        forwarder: Forwarder | None,
    ) -> None:
        self.context = context
        self.device = torch.device(context.device)

        # Keep the segment on its configured accelerator.
        self.segment = segment.to(self.device)
        if hasattr(self.segment, "eval"):
            self.segment.eval()  # type: ignore[call-arg]

        # Keep a custom forwarder colocated with the segment by default.
        self.forwarder = forwarder
        if self.forwarder is not None and getattr(self.forwarder, "device", None) is None:
            self.forwarder.device = self.device  # type: ignore[assignment]

    @torch.inference_mode()
    def run(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, ForwarderDecision | None]:
        """Process a hidden state and return forwarder decisions when available."""
        x = hidden_state.to(self.device, non_blocking=True)
        segment_output = self.segment(x)

        decision: ForwarderDecision | None = None
        if self.forwarder is not None:
            decision = self.forwarder.evaluate(segment_output)

        # Hidden states are returned to the caller rather than cached here.
        return segment_output, decision
