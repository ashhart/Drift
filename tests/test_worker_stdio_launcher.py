import json
import subprocess
import sys
import time


def launch(tmp_path, code, wall=3):
    receipt = tmp_path / 'termination.json'
    command = [sys.executable, '-m', 'drift.serving.worker_stdio_launcher', '--evidence', str(receipt),
               '--wall-seconds', str(wall), '--stop-timeout', '.1', '--', sys.executable, '-u', '-c', code]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process, receipt


def wait_receipt(path):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        try:
            result = json.loads(path.read_text())
            assert result['child_reaped'] and not result['process_group_alive']
            return result
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.02)
    raise AssertionError('termination receipt missing')


def test_stdio_eof_reaps_worker_and_preserves_private_protocol(tmp_path):
    process, receipt = launch(tmp_path, 'import sys; print(sys.stdin.readline().strip(),flush=True); sys.stdin.read()')
    process.stdin.write(b'private-test-data\n'); process.stdin.flush()
    assert process.stdout.readline() == b'private-test-data\n'
    process.stdin.close()
    assert process.wait(timeout=4) == 0
    result = wait_receipt(receipt)
    assert result['reason'] in {'child_exit', 'input_eof'}
    assert 'private-test-data' not in receipt.read_text()
    assert receipt.stat().st_mode & 0o777 == 0o600


def test_deadline_kills_ignoring_worker_and_reports_failure(tmp_path):
    process, receipt = launch(tmp_path, 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(30)', wall=1.5)
    began = time.monotonic()
    assert process.stdout.readline() == b'ready\n'
    assert process.wait(timeout=3) == 2
    result = wait_receipt(receipt)
    assert result['reason'] == 'deadline' and result['kill_sent']
    assert time.monotonic() - began < 3
    process.stdin.close()


def test_launcher_death_closes_control_and_reaps_worker(tmp_path):
    process, receipt = launch(tmp_path, 'import time; print("ready",flush=True); time.sleep(30)')
    assert process.stdout.readline() == b'ready\n'
    process.kill(); process.wait(timeout=2)
    result = wait_receipt(receipt)
    assert result['reason'] == 'controller_eof'
    process.stdin.close()


def test_reused_evidence_refuses_before_spawn(tmp_path):
    receipt = tmp_path / 'termination.json'; receipt.write_text('preserved')
    process, _ = launch(tmp_path, 'raise AssertionError("must not launch")')
    assert process.wait(timeout=2) == 2
    assert receipt.read_text() == 'preserved'
    process.stdin.close()


def test_three_minute_allowance_launches_and_reaps_without_waiting_out_budget(tmp_path):
    process, receipt = launch(tmp_path, 'import sys; print("ready",flush=True); sys.stdin.read()', wall=180)
    assert process.stdout.readline() == b'ready\n'
    process.stdin.close()
    assert process.wait(timeout=4) == 0
    assert wait_receipt(receipt)['wall_seconds'] < 4


def test_more_than_three_minutes_is_rejected_before_worker_spawn(tmp_path):
    process, receipt = launch(tmp_path, 'print("must not run",flush=True)', wall=181)
    stdout, _ = process.communicate(timeout=2)
    assert process.returncode == 2 and stdout == b'' and not receipt.exists()
