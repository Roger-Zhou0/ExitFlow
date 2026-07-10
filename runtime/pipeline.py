"""Segmented inference pipeline orchestrating router and workers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import torch

from .forwarder import ForwarderDecision
from .router import RequestRouter
from .segment_worker import SegmentWorker


@dataclass
class PipelineConfig:
    entry_segment_id: str = "segment-1"
    max_segments: int = 3


class SegmentedPipeline:
    """Coordinates the three-stage exit-enabled inference flow."""

    def __init__(
        self,
        config: PipelineConfig,
        router: RequestRouter,
        segment_chain: Iterable[SegmentWorker],
    ) -> None:
        self.config = config
        self.router = router
        self.workers: dict[str, SegmentWorker] = {
            worker.context.segment_id: worker for worker in segment_chain
        }

    def run(self, prompt: Any) -> dict[str, Any]:
        """Run the prompt through the segmented pipeline and return outputs."""
        hidden_state = self._prepare_hidden_state(prompt)
        segment_id: str | None = self.config.entry_segment_id
        segments_processed = 0
        decisions: list[ForwarderDecision] = []
        exit_segment: str | None = None

        while segment_id and segments_processed < self.config.max_segments:
            if segment_id == self.config.entry_segment_id:
                worker = self.router.next_worker()
            else:
                worker = self.workers.get(segment_id)
                if worker is None:
                    break
            hidden_state, decision = worker.run(hidden_state)
            segments_processed += 1
            if decision is None:
                exit_segment = worker.context.segment_id
                break
            decisions.append(decision)
            if decision.exit_now or decision.next_segment_id is None:
                exit_segment = worker.context.segment_id
                break
            hidden_state = decision.forwarded_state
            segment_id = decision.next_segment_id

        return {
            "decisions": decisions,
            "exit_segment": exit_segment,
            "final_hidden_state": hidden_state,
            "tokens": [],
        }

    @staticmethod
    def _prepare_hidden_state(prompt: Any) -> torch.Tensor:
        if isinstance(prompt, torch.Tensor):
            return prompt.float()
        if isinstance(prompt, (list, tuple)):
            return torch.tensor(prompt, dtype=torch.float32)
        if isinstance(prompt, str):
            encoded = [float(ord(ch)) for ch in prompt]
            if not encoded:
                encoded = [0.0]
            return torch.tensor(encoded, dtype=torch.float32)
        if isinstance(prompt, dict) and "hidden_state" in prompt:
            return SegmentedPipeline._prepare_hidden_state(prompt["hidden_state"])
        raise TypeError(f"Unsupported prompt type for segmented pipeline: {type(prompt)!r}")
