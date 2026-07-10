"""Training data pipeline definitions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SampleRecord:
    """In-memory representation of a tokenised sample along with bookkeeping."""

    input_ids: list[int]
    labels: list[int]
    size_bytes: int

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
        }


@dataclass
class TrainingDatasetConfig:
    """Configuration for preparing the KD training dataset."""

    source_path: Path
    token_budget: int
    shuffle: bool = True


def build_training_dataset(config: TrainingDatasetConfig) -> list[dict[str, list[int]]]:
    """Return tokenized samples within the configured token budget."""
    records = list(stream_training_records(config))
    samples = [record.to_dict() for record in records]
    if config.shuffle:
        random.shuffle(samples)
    return samples


def build_calibration_dataset(
    path: Path,
    *,
    token_budget: int,
    limit: Optional[int] = None,
) -> list[dict[str, list[int]]]:
    """Materialize calibration samples without shuffling."""
    config = TrainingDatasetConfig(source_path=path, token_budget=token_budget, shuffle=False)
    records = list(stream_training_records(config))
    if limit is not None:
        records = records[:limit]
    return [record.to_dict() for record in records]


def stream_training_records(config: TrainingDatasetConfig) -> Iterable[SampleRecord]:
    """Yield tokenised samples while respecting the configured token budget."""
    if not config.source_path.exists():
        raise FileNotFoundError(f"Training source not found: {config.source_path}")
    sources = _resolve_sources(config.source_path)
    return _TrainingRecordIterable(sources, config.token_budget)


def _resolve_sources(path: Path) -> list[Path]:
    if path.is_dir():
        sources = sorted(p for p in path.glob("*.jsonl") if p.is_file() or p.is_symlink())
        if not sources:
            raise FileNotFoundError(f"No JSONL files found under directory: {path}")
        return sources
    return [path]


class _TrainingRecordIterable:
    """Re-iterable view over tokenised samples on disk."""

    def __init__(self, sources: list[Path], token_budget: int) -> None:
        self._sources = sources
        self._token_budget = token_budget

    def __iter__(self):
        consumed = 0
        for source in self._sources:
            with source.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line)
                    record = SampleRecord(
                        input_ids=payload.get("input_ids", []),
                        labels=payload.get("labels", []),
                        size_bytes=len(raw_line.encode("utf-8")),
                    )
                    tokens = len(record.input_ids)
                    if consumed + tokens > self._token_budget:
                        return
                    consumed += tokens
                    yield record
