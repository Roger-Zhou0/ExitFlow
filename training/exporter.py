"""Artifact export helpers for trained exit heads and backbone."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from training.kd_trainer import ExitHeadSpec


@dataclass
class ExportPaths:
    backbone_dir: Path
    exit_head_dir: Path
    metadata_path: Path


class ArtifactExporter:
    """Exports trained modules to disk for runtime consumption."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = ExportPaths(
            backbone_dir=root / "backbone",
            exit_head_dir=root / "exit_heads",
            metadata_path=root / "metadata.json",
        )
        self.paths.backbone_dir.mkdir(parents=True, exist_ok=True)
        self.paths.exit_head_dir.mkdir(parents=True, exist_ok=True)

    def export_backbone(self, backbone: torch.nn.Module) -> None:
        """Persist the backbone parameters using the best available method."""
        if hasattr(backbone, "save_pretrained"):
            backbone.save_pretrained(str(self.paths.backbone_dir))
        else:
            torch.save(backbone.state_dict(), self.paths.backbone_dir / "backbone.pt")

    def export_exit_heads(self, exits: Iterable[ExitHeadSpec]) -> dict[str, str]:
        """Export each exit head as TorchScript modules."""
        exported: dict[str, str] = {}
        for spec in exits:
            try:
                script = torch.jit.script(spec.head)
            except Exception:
                example = torch.zeros(1, spec.hidden_size)
                script = torch.jit.trace(spec.head, example)
            target_path = self.paths.exit_head_dir / f"{spec.identifier}.pt"
            script.save(str(target_path))
            exported[spec.identifier] = str(target_path)
        return exported

    def write_metadata(self, exported_exit_heads: dict[str, str]) -> None:
        """Record manifest-friendly artifact locations."""
        metadata = {
            "backbone_dir": str(self.paths.backbone_dir),
            "exit_heads": exported_exit_heads,
        }
        with self.paths.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
