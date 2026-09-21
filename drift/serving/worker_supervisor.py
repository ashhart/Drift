"""Own a local worker process group and emit aggregate termination evidence only."""
import argparse
import json
import math
import os
import signal
import subprocess
import threading
import time
from drift.serving.worker_supervisor_io import RelayStop, relay


def group_alive(pid):
    try: os.killpg(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def stop_group(child, timeout, graceful=False):
    term = killed = False
    if child.stdin is not None:
        child.stdin.close()
    if graceful:
        try: child.wait(timeout=timeout)
        except subprocess.TimeoutExpired: pass
    if group_alive(child.pid):
        term = True
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: child.wait(timeout=timeout)
    except subprocess.TimeoutExpired: pass
    if group_alive(child.pid):
        killed = True
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    try: child.wait(timeout=timeout)
    except subprocess.TimeoutExpired: pass
    child.stdout.close()
    return term, killed


def supervise_worker(command, *, control_fd, evidence_fd, wall_seconds, input_fd=None, output_fd=None,
                     activation_fd=None, stop_timeout=2, max_pending_bytes=262144, max_total_bytes=16777216):
    limits = (wall_seconds, stop_timeout)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in limits):
        raise ValueError('INVALID_LIMIT')
    if wall_seconds <= 3 * stop_timeout + 0.25:
        raise ValueError('INVALID_LIMIT')
    if any(type(value) is not int or value <= 0 for value in (max_pending_bytes, max_total_bytes)):
        raise ValueError('INVALID_LIMIT')
    fds = [fd for fd in (control_fd, evidence_fd, input_fd, output_fd, activation_fd) if fd is not None]
    if any(type(fd) is not int or fd < 0 for fd in fds) or len(set(fds)) != len(fds) or control_fd < 3 or evidence_fd < 3:
        raise ValueError('INVALID_FDS')
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('SUPERVISOR_REQUIRES_MAIN_THREAD')
    started = time.monotonic(); stopped = threading.Event(); previous = {}
    child = None
    counts = {'input_bytes': 0, 'output_bytes': 0}
    reason = 'launch_failure'
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *args: stopped.set())
        child = subprocess.Popen(list(command), stdin=subprocess.PIPE if input_fd is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True,
                                 pass_fds=() if activation_fd is None else (activation_fd,))
        if activation_fd is not None:
            os.close(activation_fd)
        try:
            deadline = started + wall_seconds - 3 * stop_timeout - 0.25
            reason = relay(child, control_fd, input_fd, output_fd, deadline, stopped.is_set,
                           max_pending_bytes, max_total_bytes, counts)
        except RelayStop as error:
            reason = str(error)
        except Exception:
            reason = 'supervisor_io_failure'
    finally:
        try:
            if child is not None:
                term, killed = stop_group(child, stop_timeout, graceful=reason == 'input_eof')
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
    if child is None:
        raise RuntimeError('WORKER_LAUNCH_FAILED')
    if child.poll() is None:
        reason = 'cleanup_timeout'
    result = {'event': 'worker_terminated', 'reason': reason, 'child_pid': child.pid,
              'returncode': child.returncode, 'child_reaped': child.poll() is not None,
              'process_group_alive': group_alive(child.pid), 'term_sent': term, 'kill_sent': killed,
              'wall_seconds': time.monotonic() - started, **counts}
    raw = json.dumps(result, allow_nan=False).encode() + b'\n'
    os.set_blocking(evidence_fd, False)
    try:
        if os.write(evidence_fd, raw) != len(raw): raise OSError
    except OSError:
        raise RuntimeError('TERMINATION_EVIDENCE_UNDELIVERED') from None
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('control', 'evidence', 'input', 'output', 'activation'):
        parser.add_argument('--' + name + '-fd', type=int, required=name in ('control', 'evidence'))
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--stop-timeout', type=float, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    if args['command'][:1] == ['--']: args['command'] = args['command'][1:]
    try:
        receipt = supervise_worker(**args)
        return 0 if receipt['reason'] == 'child_exit' and receipt['returncode'] == 0 and not receipt['process_group_alive'] else 2
    except Exception:
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
