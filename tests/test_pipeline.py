from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from config.loader import load_manifest, load_thresholds
from runtime import RuntimeFactories, build_segmented_pipeline


class IdentityModule(nn.Module):
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state


def _build_pipeline():
    manifest = deepcopy(load_manifest(Path("config/manifest.json")))
    thresholds = load_thresholds(Path("config/thresholds.json"))
    manifest["runtime"].pop("native_backbone_dir", None)
    manifest["runtime"].pop("segment_pools", None)
    for worker in manifest["runtime"]["workers"].values():
        worker["device"] = "cpu"
    factories = RuntimeFactories(
        segment_factory=lambda _: IdentityModule(),
        exit_head_factory=lambda _: IdentityModule(),
    )
    return build_segmented_pipeline(manifest, thresholds, factories)


def test_pipeline_exits_in_segment_one():
    pipeline = _build_pipeline()
    prompt = torch.tensor([2.0, 1.8, 1.6], dtype=torch.float32)
    result = pipeline.run(prompt)
    assert result["exit_segment"] == "segment-1"
    assert result["final_hidden_state"] is not None


def test_pipeline_falls_through_to_third_segment():
    pipeline = _build_pipeline()
    prompt = torch.tensor([0.2, 0.1, -0.1], dtype=torch.float32)
    result = pipeline.run(prompt)
    assert result["exit_segment"] == "segment-8"
