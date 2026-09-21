"""Keep qualification verdicts strict on stale readiness and CLI recovery."""
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('boundary_verdict', Path(__file__).parents[1] / 'scripts/omp/boundary_probe_verdict.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def observation(mode='release'):
    worker = dict(terminal=1, stream=1, close=1)
    return dict(mode=mode, events=[dict(kind='tool_call', terminals=1, streams=1, pid=1), dict(kind='admitted', tool='hub', at=20, pid=1)], parked=dict(workers={'parent': worker, 'child': worker}), workers={'parent': worker, 'child': worker}, released_at=10, ready_remaining=[], exit_code=0)


def test_bound_release_after_completed_generation_qualifies():
    assert MODULE.assess(observation())['verdict'] == 'PASSED'


def test_stale_child_ready_does_not_qualify_even_with_exit_zero():
    report = observation('timeout')
    report['parked']['workers']['child'] = dict(terminal=1, stream=2)
    report['ready_remaining'] = ['child']
    assert MODULE.assess(report)['verdict'] == 'FAILED'


def test_hook_before_terminal_is_not_a_native_safe_boundary():
    report = observation()
    report['events'][0]['terminals'] = 0
    assert MODULE.assess(report)['verdict'] == 'FAILED'


def test_blocked_tool_does_not_hide_failed_readiness_cleanup():
    report = observation('invalid')
    report['events'].pop()
    report['ready_remaining'] = ['parent', 'child']
    assert MODULE.assess(report)['verdict'] == 'FAILED'
    report['ready_remaining'] = []
    assert MODULE.assess(report)['verdict'] == 'PASSED'


def test_cancel_cannot_pass_without_both_worker_cleanup_receipts():
    report = observation('cancel'); report['events'].pop()
    report['workers']['child'] = dict(terminal=1, stream=1, close=0)
    assert MODULE.assess(report)['verdict'] == 'FAILED'


def test_missing_worker_metadata_is_not_a_completed_pair():
    report = observation(); report['workers'] = {}
    report['parked']['workers'] = {}
    assert MODULE.assess(report)['verdict'] == 'FAILED'
