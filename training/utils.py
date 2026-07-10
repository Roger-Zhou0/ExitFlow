"""Shared helpers for training routines."""

from __future__ import annotations

from typing import List


def batch_samples(samples: List[dict[str, list[int]]], batch_size: int) -> List[dict[str, list[list[int]]]]:
    """Group sequential samples into batches of size ``batch_size``."""
    if batch_size <= 1:
        return samples
    batched: List[dict[str, list[list[int]]]] = []
    for offset in range(0, len(samples), batch_size):
        chunk = samples[offset : offset + batch_size]
        batched.append(
            {
                "input_ids": [sample["input_ids"] for sample in chunk],
                "labels": [sample["labels"] for sample in chunk],
            }
        )
    return batched
