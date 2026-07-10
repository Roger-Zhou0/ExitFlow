"""Evaluation runner for segmented inference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, List

import torch

from runtime.pipeline import SegmentedPipeline

from .benchmarks import BenchmarkSuite
from .metrics import MetricTracker

BenchmarkScorer = Callable[[dict, dict], float]


@dataclass
class EvaluationConfig:
    max_tokens: int = 256
    progress_interval: int = 50
    verbose: bool = False


class EvaluationRunner:
    """Executes evaluation benchmarks and logs relevant metrics."""

    def __init__(
        self,
        config: EvaluationConfig,
        benchmarks: BenchmarkSuite,
        tracker: MetricTracker,
        pipeline: SegmentedPipeline,
        segment_order: Iterable[str],
        scorers: Dict[str, BenchmarkScorer] | None = None,
    ) -> None:
        self.config = config
        self.benchmarks = benchmarks
        self.tracker = tracker
        self.pipeline = pipeline
        self.segment_order = list(segment_order)
        self.scorers = scorers or {}

    def run_all(self) -> None:
        for benchmark in self.benchmarks:
            samples = benchmark.load_samples()
            if self.config.verbose:
                print(f"[bench] {benchmark.name}: {len(samples)} samples")
            self._run_single(benchmark.name, samples)

    def _run_single(self, name: str, samples: List[dict]) -> None:
        latencies: List[float] = []
        processed = 0
        for sample in samples:
            if "input_ids" in sample:
                prompt = sample["input_ids"]
            else:
                prompt = sample.get("prompt")
            if prompt is None:
                continue
            start = perf_counter()
            result = self.pipeline.run(prompt)
            latency_ms = (perf_counter() - start) * 1000
            latencies.append(latency_ms)
            exit_depth = self._compute_exit_depth(result)
            if exit_depth is not None:
                self.tracker.log("avg_exit_depth", exit_depth)
            self._score_sample(sample, result)
            processed += 1
            if self.config.verbose and processed % max(1, self.config.progress_interval) == 0:
                print(
                    f"[bench:{name}] processed={processed}/{len(samples)} last_latency_ms={latency_ms:.2f}"
                )
        if latencies:
            p95 = float(torch.quantile(torch.tensor(latencies), torch.tensor(0.95)))
            self.tracker.log("p95_latency", p95)
            if self.config.verbose:
                print(f"[bench:{name}] p95_latency={p95:.2f}ms samples={processed}")

    def _compute_exit_depth(self, result: dict) -> float | None:
        exit_segment = result.get("exit_segment")
        if exit_segment is None:
            return None
        try:
            index = self.segment_order.index(exit_segment)
        except ValueError:
            return None
        return float(index + 1)

    def _score_sample(self, sample: dict, result: dict) -> None:
        for metric, scorer in self.scorers.items():
            value = scorer(sample, result)
            self.tracker.log(metric, float(value))
