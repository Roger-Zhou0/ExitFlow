from pathlib import Path

import torch
from torch import nn

from training.checkpoint import CheckpointManager
from training.kd_trainer import ExitHeadSpec
from training.utils import batch_samples


def test_checkpoint_roundtrip(tmp_path: Path):
    manager = CheckpointManager(tmp_path, keep=2)
    backbone = nn.Linear(4, 4)
    exit_head = nn.Linear(4, 3)
    spec = ExitHeadSpec(identifier="forwarder@12", layer_index=12, hidden_size=4, head=exit_head)
    optimizer = torch.optim.Adam(exit_head.parameters(), lr=1e-3)

    manager.save(
        epoch=0,
        sample_index=0,
        global_step=1,
        accumulation_offset=0,
        backbone=backbone,
        exits=[spec],
        optimizer=optimizer,
    )

    exit_head.weight.data.zero_()
    state = manager.restore(backbone=backbone, exits=[spec], optimizer=optimizer)

    assert state is not None
    assert state.global_step == 1
    assert not torch.allclose(exit_head.weight.data, torch.zeros_like(exit_head.weight.data))


def test_batch_samples_groups_records():
    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
        {"input_ids": [7, 8, 9], "labels": [7, 8, 9]},
    ]
    batched = batch_samples(samples, batch_size=2)
    assert len(batched) == 2
    assert batched[0]["input_ids"][0] == [1, 2, 3]
    assert batched[0]["input_ids"][1] == [4, 5, 6]
