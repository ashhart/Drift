"""Keep inference admission and cancellation recovery tied to observed engine state."""
import pytest

from drift.serving.glm_guard import EngineGuard
from drift.serving.live_metrics import MetricsSnapshot


def snapshot(running=0, waiting=0, kv=0):
    return MetricsSnapshot(running, waiting, kv, 'fixture', '0')


def test_busy_engine_is_rejected_before_a_new_owned_request():
    guard = EngineGuard(lambda timeout: snapshot(running=1))
    with pytest.raises(ValueError):
        guard.admit(2)


def test_recovery_requires_three_consecutive_idle_baseline_samples():
    states = iter([snapshot(), snapshot(running=1, kv=.1), snapshot(), snapshot(), snapshot()])
    calls = []
    def sample(timeout):
        calls.append(timeout)
        return next(states)
    guard = EngineGuard(sample, pause=lambda value: None)
    guard.admit(2)
    guard.recover()
    assert len(calls) == 5


def test_recovery_without_an_admitted_inference_is_a_noop():
    guard = EngineGuard(lambda timeout: pytest.fail('no model request was made'))
    guard.recover()
