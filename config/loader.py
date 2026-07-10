"""Utilities for loading manifest and threshold configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class ForwarderConfigDict(TypedDict, total=False):
    id: str
    exit_head_weights: str
    forwarder_weights: str
    next_segment: str | None


class ManifestModelSegment(TypedDict, total=False):
    id: str
    layers: dict[str, int]
    components: list[str]
    forwarder: ForwarderConfigDict | None
    artifacts: dict[str, str]


class Manifest(TypedDict):
    project: dict[str, Any]
    model: dict[str, Any]
    runtime: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]


@dataclass
class ConfigPaths:
    manifest_path: Path
    thresholds_path: Path


def load_manifest(path: Path) -> Manifest:
    """Load the manifest JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    """Load the thresholds JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
