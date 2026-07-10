"""Metric tracking for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class MetricTracker:
    """Accumulates metrics such as accuracy and latency."""

    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, name: str, value: float) -> None:
        self.metrics.setdefault(name, []).append(value)

    def average(self, name: str) -> float:
        values = self.metrics.get(name, [])
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    def percentile(self, name: str, q: float) -> float:
        values = self.metrics.get(name, [])
        if not values:
            return float("nan")
        tensor = torch.tensor(values)
        return float(torch.quantile(tensor, torch.tensor(q)))

    def summary(self) -> dict[str, float]:
        return {name: self.average(name) for name in self.metrics}
