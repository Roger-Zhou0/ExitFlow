"""Evaluation package for benchmarking segmented inference."""

from .benchmarks import Benchmark, BenchmarkSuite
from .metrics import MetricTracker
from .runner import EvaluationConfig, EvaluationRunner
from .scorers import build_default_scorers, compliance_score, exit_segment_accuracy

__all__ = [
    "Benchmark",
    "BenchmarkSuite",
    "EvaluationConfig",
    "EvaluationRunner",
    "MetricTracker",
    "build_default_scorers",
    "compliance_score",
    "exit_segment_accuracy",
]
