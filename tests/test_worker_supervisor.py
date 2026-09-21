import json
import os
import signal
import sys
import time
import pytest
from drift.serving.worker_supervisor import supervise_worker


def run_child(source, **kwargs):
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe()
    try:
        result = supervise_worker([sys.executable, '-c', source], control_fd=control_r, evidence_fd=evidence_w,
                                   wall_seconds=kwargs.pop('wall_seconds', 2), stop_timeout=0.1, **kwargs)
        assert json.loads(os.read(evidence_r, 4096)) == result
        return result
    finally:
        for fd in (control_r, control_w, evidence_r, evidence_w): os.close(fd)


def test_standalone_child_is_reaped_without_retaining_output():
    result = run_child("print('synthetic private output')")
    assert result['reason'] == 'child_exit' and result['returncode'] == 0
    assert result['child_reaped'] and not result['process_group_alive']
    assert result['output_bytes'] > 0 and 'synthetic' not in repr(result)


def test_stuck_child_is_killed_within_declared_wall_budget():
    result = run_child('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)', wall_seconds=1)
    assert result['reason'] == 'deadline' and result['kill_sent']
    assert result['child_reaped'] and result['wall_seconds'] < 1.2
    with pytest.raises(ProcessLookupError): os.kill(result['child_pid'], 0)


def test_dropped_controller_stops_its_owned_child():
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe(); os.close(control_w)
    try:
        result = supervise_worker([sys.executable, '-c', 'import time; time.sleep(20)'], control_fd=control_r,
                                  evidence_fd=evidence_w, wall_seconds=2, stop_timeout=0.1)
        assert result['reason'] == 'controller_eof' and result['child_reaped']
    finally:
        for fd in (control_r, evidence_r, evidence_w): os.close(fd)


def test_output_budget_failure_cannot_leave_child_alive():
    result = run_child("import time; print('x'*20000, flush=True); time.sleep(20)", max_total_bytes=1000)
    assert result['reason'] == 'io_limit' and result['child_reaped']


def test_supervisor_cli_relays_private_bytes_and_emits_separate_receipt(tmp_path):
    import select
    import subprocess
    child = tmp_path / 'child.py'
    child.write_text("import sys\nfor line in sys.stdin:\n print(line.strip(), flush=True)\n if line.strip() == 'close': break\n")
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe()
    own_r, own_w = os.pipe(); out_r, out_w = os.pipe()
    fds = (control_r, evidence_w, own_r, out_w)
    process = subprocess.Popen([sys.executable, '-m', 'drift.serving.worker_supervisor', '--control-fd', str(control_r), '--evidence-fd', str(evidence_w), '--input-fd', str(own_r), '--output-fd', str(out_w), '--wall-seconds', '3', '--stop-timeout', '0.1', '--', sys.executable, str(child)], pass_fds=fds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        os.write(own_w, b'private fixture\n')
        assert select.select([out_r], [], [], 2)[0]
        assert os.read(out_r, 4096) == b'private fixture\n'
        os.write(own_w, b'close\n')
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0 and stdout == stderr == b''
        receipt = json.loads(os.read(evidence_r, 4096))
        assert receipt['child_reaped'] and receipt['reason'] == 'child_exit'
        assert 'private fixture' not in repr(receipt)
    finally:
        if process.poll() is None: process.kill(); process.wait()
        for fd in (control_r, control_w, evidence_r, evidence_w, own_r, own_w, out_r, out_w): os.close(fd)


def test_explicit_abort_is_distinct_from_completion():
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe()
    os.write(control_w, b'{"op":"abort"}\n')
    try:
        receipt = supervise_worker([sys.executable, '-c', 'import time; time.sleep(20)'], control_fd=control_r,
                                   evidence_fd=evidence_w, wall_seconds=2, stop_timeout=0.1)
        assert receipt['reason'] == 'control_abort' and receipt['child_reaped']
    finally:
        for fd in (control_r, control_w, evidence_r, evidence_w): os.close(fd)
