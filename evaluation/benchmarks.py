"""Benchmark definitions for accuracy and latency tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List


@dataclass
class Benchmark:
    name: str
    dataset_path: Path

    def load_samples(self) -> List[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset missing: {self.dataset_path}")
        records: List[dict] = []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records


class BenchmarkSuite:
    """Maintains the list of benchmarks used for evaluation."""

    def __init__(self, benchmarks: Iterable[Benchmark]) -> None:
        self.benchmarks = list(benchmarks)

    def __iter__(self) -> Iterator[Benchmark]:
        return iter(self.benchmarks)
