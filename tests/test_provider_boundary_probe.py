"""Reject incomplete provider-boundary evidence and independently reap probe children."""
import importlib
from pathlib import Path
import sys
import time

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/omp'))


def evidence(mode='timeout'):
    worker = dict(terminal=2, stream=2, close=1)
    return dict(mode=mode, events=[dict(kind='tool_call', terminals=2, streams=2, pid=1), dict(kind='admitted', tool='hub', at=30, pid=1)], parked=dict(workers={'parent': worker, 'child': worker}), workers={'parent': worker, 'child': worker}, released_at=20, ready_remaining=[], failed_roles=[], exit_code=0, cleanup=dict(reaped=True, joined=True, group_gone=True, worker_pids_gone=True, terminated=False, output_limit=False), abort_sent=False, abort_ack=False)


def test_short_tool_timeout_still_requires_valid_release():
    assess = importlib.import_module('provider_boundary_verdict').assess
    assert assess(evidence())['verdict'] == 'PASSED'
    report = evidence(); report['released_at'] = None
    assert assess(report)['verdict'] == 'FAILED'


@pytest.mark.parametrize('field', ['reaped', 'joined', 'group_gone', 'worker_pids_gone'])
def test_cleanup_cannot_be_assumed(field):
    assess = importlib.import_module('provider_boundary_verdict').assess
    report = evidence(); report['cleanup'][field] = False
    assert assess(report)['verdict'] == 'FAILED'


def test_abort_requires_real_ack_and_invalidation_without_admission():
    assess = importlib.import_module('provider_boundary_verdict').assess
    report = evidence('abort'); report['events'].pop(); report['failed_roles'] = ['parent', 'child']
    assert assess(report)['verdict'] == 'FAILED'
    report.update(abort_sent=True, abort_ack=True)
    assert assess(report)['verdict'] == 'PASSED'
    report['cleanup']['terminated'] = True
    assert assess(report)['verdict'] == 'FAILED'


def test_invalid_release_does_not_count_cli_recovery_as_success():
    assess = importlib.import_module('provider_boundary_verdict').assess
    report = evidence('invalid'); report['failed_roles'] = ['parent', 'child']
    assert assess(report)['verdict'] == 'FAILED'
    report['events'].pop()
    assert assess(report)['verdict'] == 'PASSED'
    report['ready_remaining'] = ['child']
    assert assess(report)['verdict'] == 'FAILED'


def test_probe_reaps_stalled_child_without_pipe_deadlock():
    Process = importlib.import_module('provider_probe_process').ProbeProcess
    actor = Process([sys.executable, '-c', 'import os,time;os.write(1,b"x"*131072);time.sleep(10)'], Path.cwd(), None)
    started = time.monotonic()
    receipt = actor.close(time.monotonic() + 2)
    assert receipt['reaped'] and receipt['joined'] and receipt['group_gone']
    assert receipt['terminated'] and time.monotonic() - started < 2.5


def test_poisoned_workers_need_independent_group_cleanup_not_graceful_callbacks():
    assess = importlib.import_module('provider_boundary_verdict').assess
    report = evidence('invalid'); report['events'].pop(); report['failed_roles'] = ['parent', 'child']
    for worker in report['workers'].values(): worker['close'] = 0
    result = assess(report)
    assert result['verdict'] == 'BLOCKED' and not result['both_workers_closed']
    assert result['boundary_verdict'] == result['process_cleanup_verdict'] == 'PASSED'
    assert result['graceful_close_verdict'] == 'BLOCKED'
    report['cleanup']['group_gone'] = False
    assert assess(report)['verdict'] == 'FAILED'


def test_rpc_ack_is_explicit_and_public_stream_content_is_discarded():
    Process = importlib.import_module('provider_probe_process').ProbeProcess
    code = 'import json,sys;json.loads(sys.stdin.readline());print(json.dumps(dict(type="response",command="abort",id="probe-abort",success=True,private="never retain")),flush=True);print(json.dumps(dict(type="agent_end")),flush=True)'
    actor = Process([sys.executable, '-c', code], Path.cwd(), None, rpc=True)
    actor.send(dict(type='abort'))
    actor.process.wait(timeout=2)
    receipt = actor.close(time.monotonic() + 2)
    assert actor.abort_ack and actor.agent_end and not receipt['terminated']
    assert receipt['reaped'] and receipt['group_gone'] and receipt['joined']
    assert 'never retain' not in repr(receipt)


def test_rpc_false_or_unrelated_ack_does_not_qualify():
    Process = importlib.import_module('provider_probe_process').ProbeProcess
    code = 'import json;print(json.dumps(dict(type="response",command="abort",id="other",success=True)));print(json.dumps(dict(type="response",command="abort",id="probe-abort",success=False)))'
    actor = Process([sys.executable, '-c', code], Path.cwd(), None, rpc=True)
    actor.process.wait(timeout=2)
    actor.close(time.monotonic() + 2)
    assert not actor.abort_ack


@pytest.mark.parametrize('mode', ['timeout', 'abort', 'invalid', 'deadline'])
def test_missing_graceful_close_never_reaches_passed(mode):
    assess = importlib.import_module('provider_boundary_verdict').assess
    report = evidence(mode)
    if mode != 'timeout':
        report['events'].pop(); report['failed_roles'] = ['parent', 'child']
        report.update(abort_sent=True, abort_ack=True)
    assert assess(report)['verdict'] == 'PASSED'
    report['workers']['child'] = dict(report['workers']['child'], close=0)
    downgraded = assess(report)
    assert downgraded['graceful_close_verdict'] == 'BLOCKED'
    assert downgraded['both_workers_closed'] is False
    assert downgraded['verdict'] == 'BLOCKED'
    assert downgraded['process_cleanup_verdict'] == 'PASSED'
