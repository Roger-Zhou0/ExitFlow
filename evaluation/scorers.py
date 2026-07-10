"""Default scoring utilities for evaluation benchmarks."""

from __future__ import annotations

from typing import Callable, Dict


def exit_segment_accuracy(sample: dict, result: dict) -> float:
    expected = sample.get("expected_exit_segment")
    observed = result.get("exit_segment")
    if expected is None:
        return 0.0
    return 1.0 if expected == observed else 0.0


def compliance_score(sample: dict, result: dict) -> float:
    del result
    return float(sample.get("compliance", 1.0))


def build_default_scorers() -> Dict[str, Callable[[dict, dict], float]]:
    return {
        "accuracy": exit_segment_accuracy,
        "compliance": compliance_score,
    }


__all__ = [
    "build_default_scorers",
    "compliance_score",
    "exit_segment_accuracy",
]
