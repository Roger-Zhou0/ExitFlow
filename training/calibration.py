"""Threshold calibration utilities for forwarder gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class CalibrationSample:
    logits: list[float]
    label: int


@dataclass
class CalibrationConfig:
    temperature: float
    target_percentile: float = 0.9


class ThresholdCalibrator:
    """Computes threshold settings for forwarders using held-out data."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    def calibrate(self, samples: Iterable[CalibrationSample]) -> dict[str, float]:
        """Return calibrated thresholds keyed by metric name."""
        top_logits: list[float] = []
        margins: list[float] = []
        entropies: list[float] = []
        for sample in samples:
            logits = torch.tensor(sample.logits)
            scaled = logits / self.config.temperature
            probs = torch.softmax(scaled, dim=-1)
            top_values = torch.topk(scaled, k=min(2, scaled.shape[-1])).values
            top_logits.append(top_values[0].item())
            if top_values.numel() > 1:
                margins.append((top_values[0] - top_values[1]).item())
            else:
                margins.append(float("inf"))
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            entropies.append(entropy)
        if not top_logits:
            raise ValueError("No calibration samples provided")
        percentile = self.config.target_percentile
        logit_threshold = torch.quantile(torch.tensor(top_logits), torch.tensor(percentile)).item()
        margin_gate = torch.quantile(torch.tensor(margins), torch.tensor(percentile)).item()
        entropy_gate = torch.quantile(torch.tensor(entropies), torch.tensor(1 - percentile)).item()
        return {
            "logit_threshold": logit_threshold,
            "margin_gate": margin_gate,
            "entropy_gate": entropy_gate,
            "temperature_scale": self.config.temperature,
        }
