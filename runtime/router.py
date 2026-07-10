"""Request routing for segmented inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

from .segment_worker import SegmentWorker


@dataclass
class RouterConfig:
    strategy: str
    segment_1_pool_size: int


class RequestRouter:
    """Distributes incoming prompts to segment-1 workers."""

    def __init__(self, config: RouterConfig, workers: Iterable[SegmentWorker]) -> None:
        self.config = config
        self.pool: Deque[SegmentWorker] = deque(workers)

    def next_worker(self) -> SegmentWorker:
        """Return the next worker according to the configured strategy."""
        if not self.pool:
            raise RuntimeError("No segment-1 workers registered")
        worker = self.pool.popleft()
        self.pool.append(worker)
        return worker
