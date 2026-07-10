import torch
import pytest

from runtime.forwarder import Forwarder, ForwarderConfig


class DummyExitHead:
    def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([[0.1, 0.9, 2.5]])
        return logits


def test_forwarder_triggers_exit():
    forwarder = Forwarder(
        DummyExitHead(),
        ForwarderConfig(
            logit_threshold=2.0,
            margin_gate=1.0,
            entropy_gate=1.0,
            temperature_scale=1.0,
            next_segment_id="segment-2",
        ),
    )
    hidden_state = torch.zeros((1, 3))
    decision = forwarder.evaluate(hidden_state)
    assert decision.exit_now is True
    assert decision.next_segment_id is None
    assert decision.forwarded_state is None


def test_forwarder_forwards_when_below_threshold():
    forwarder = Forwarder(
        DummyExitHead(),
        ForwarderConfig(
            logit_threshold=3.0,
            margin_gate=2.0,
            entropy_gate=0.5,
            temperature_scale=1.0,
            next_segment_id="segment-2",
        ),
    )
    hidden_state = torch.zeros((1, 3))
    decision = forwarder.evaluate(hidden_state)
    assert decision.exit_now is False
    assert decision.next_segment_id == "segment-2"
    assert decision.forwarded_state is hidden_state


def test_forwarder_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature_scale"):
        Forwarder(
            DummyExitHead(),
            ForwarderConfig(
                logit_threshold=2.0,
                margin_gate=1.0,
                entropy_gate=1.0,
                temperature_scale=0.0,
                next_segment_id="segment-2",
            ),
        )
