"""Cancellation evidence is sampled with fake time and synthetic owned-process state."""
from types import SimpleNamespace

import pytest

from drift.serving.live_cancel_observation import CancellationObserver
from drift.serving.live_metrics import MetricsError, MetricsSnapshot

MODEL = 'GLM-5.3-Flash-EXL3'


def state(**changes):
    return SimpleNamespace(**{'started': False, 'active': False, 'abort_sent': False,
                              'cancelled': False, 'completed': False, 'concurrent': False, **changes})


class Trial:
    def __init__(self, **kwargs):
        self.now = 0.0
        self.observer = CancellationObserver(clock=lambda: self.now, **kwargs)
        self.timeouts = []

    def poll(self, values=(0, 0, 0), evidence=None, delay=0):
        self.now += 1
        def sample(*, timeout):
            self.timeouts.append(timeout)
            self.now += delay
            return MetricsSnapshot(*values, MODEL, '0')
        return self.observer.poll(sample, lambda: evidence or state())

    def ready(self):
        for _ in range(3): result = self.poll((0, 0, 0.1))
        assert result['action'] == 'start'

    def active(self):
        result = self.poll((1, 0, 0.3), state(started=True, active=True))
        assert result['action'] == 'abort'


def test_requires_baseline_activity_abort_exit_and_three_settled_samples():
    trial = Trial()
    trial.ready()
    trial.active()
    evidence = state(started=True, abort_sent=True, cancelled=True)
    assert trial.poll((0, 0, 0.2), evidence)['status'] == 'RUNNING'
    assert trial.poll((0, 0, 0.1), evidence)['status'] == 'RUNNING'
    assert trial.poll((0, 0, 0.09), evidence)['status'] == 'RUNNING'
    result = trial.poll((0, 0, 0.1), evidence)
    assert result['status'] == 'PASSED'
    assert result['settled_samples'] == 3
    assert result['baseline']['kv_usage'] == 0.1
    assert trial.timeouts == sorted(trial.timeouts, reverse=True)


@pytest.mark.parametrize('evidence,reason', [
    (state(started=True, completed=True), 'NATURAL_COMPLETION'),
    (state(started=True, abort_sent=True, cancelled=True), 'ABORT_BEFORE_ACTIVITY'),
    (state(concurrent=True), 'CONCURRENT_ACTIVITY'),
])
def test_invalid_evidence_cannot_qualify(evidence, reason):
    trial = Trial()
    trial.ready()
    result = trial.poll(evidence=evidence)
    assert result['status'] == 'INVALID' and result['reason'] == reason


def test_metrics_missing_and_late_samples_do_not_pass():
    trial = Trial(timeout=5)
    assert trial.poll(delay=5)['reason'] == 'DEADLINE_EXCEEDED'
    trial = Trial()
    def bad_sample(**kwargs):
        raise MetricsError('synthetic private metric payload')
    result = trial.observer.poll(bad_sample, state)
    assert result['status'] == 'INVALID' and result['reason'] == 'METRICS_UNAVAILABLE'
    assert 'synthetic private' not in str(result)


def test_missing_local_exit_and_concurrent_metrics_are_distinguished():
    trial = Trial()
    trial.ready()
    trial.active()
    for _ in range(3):
        assert trial.poll(evidence=state(started=True, abort_sent=True))['status'] == 'RUNNING'
    assert trial.poll((2, 0, 0), state(started=True, abort_sent=True))['reason'] == 'CONCURRENT_ACTIVITY'


def test_idle_baseline_does_not_accept_a_busy_engine():
    trial = Trial()
    result = trial.poll((1, 0, 0.1))
    assert result['status'] == 'INVALID' and result['reason'] == 'BASELINE_NOT_IDLE'


def test_settled_samples_must_be_consecutive_and_after_local_exit():
    trial = Trial()
    trial.ready()
    trial.active()
    evidence = state(started=True, abort_sent=True, cancelled=True)
    assert trial.poll((0, 0, 0.1), evidence)['settled_samples'] == 1
    assert trial.poll((0, 0, 0.2), evidence)['settled_samples'] == 0
    assert trial.poll((0, 0, 0.1), evidence)['settled_samples'] == 1
    assert trial.poll((0, 0, 0.1), evidence)['settled_samples'] == 2
    assert trial.poll((0, 0, 0.1), evidence)['status'] == 'PASSED'


def test_same_clock_reading_does_not_manufacture_multiple_samples():
    observer = CancellationObserver(clock=lambda: 0.0, max_samples=4)
    sample = lambda **kwargs: MetricsSnapshot(0, 0, 0, MODEL, '0')
    for _ in range(4):
        result = observer.poll(sample, state)
        assert result['baseline_samples'] == 1 and result['action'] is None
    assert observer.poll(sample, state)['reason'] == 'SAMPLE_LIMIT_EXCEEDED'


def test_local_state_callback_cannot_complete_after_deadline():
    trial = Trial(timeout=5)
    def late_state():
        trial.now = 5
        return state()
    result = trial.observer.poll(lambda **kwargs: MetricsSnapshot(0, 0, 0, MODEL, '0'), late_state)
    assert result['status'] == 'FAILED' and result['reason'] == 'DEADLINE_EXCEEDED'
    assert result['remaining_seconds'] == 0 and result['elapsed_seconds'] == 5


@pytest.mark.parametrize('problem', ['target_changed', 'state_missing', 'abort_missing', 'started_early'])
def test_wrong_target_and_missing_or_premature_evidence_are_invalid(problem):
    trial = Trial()
    if problem == 'started_early':
        result = trial.poll(evidence=state(started=True))
    else:
        trial.ready()
        if problem == 'target_changed':
            result = trial.observer.poll(lambda **kwargs: MetricsSnapshot(0, 0, 0, 'other', '0'), state)
        elif problem == 'state_missing':
            result = trial.observer.poll(lambda **kwargs: MetricsSnapshot(0, 0, 0, MODEL, '0'), lambda: None)
        else:
            result = trial.poll(evidence=state(started=True, cancelled=True))
    assert result['status'] == 'INVALID'


def test_terminal_result_is_latched_without_fetching_more_metrics():
    trial = Trial(timeout=1)
    assert trial.poll()['reason'] == 'DEADLINE_EXCEEDED'
    def forbidden(**kwargs):
        pytest.fail('terminal observation sampled again')
    assert trial.observer.poll(forbidden, forbidden)['status'] == 'FAILED'
