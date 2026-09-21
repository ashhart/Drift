"""Expose a bounded owner activation socket alongside an unchanged SSH stdio worker."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import threading
import time
from drift.serving.worker_activation_files import private_root
from drift.serving.worker_broker import launch_owner_worker
from drift.serving.worker_owner_socket import OwnerSocket
from drift.serving.worker_supervisor_io import RelayStop, relay


def pinned_json(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode): raise ValueError('OWNER_FILE')
        raw = handle.read(1048577)
    if len(raw) > 1048576 or hashlib.sha256(raw).hexdigest() != expected: raise ValueError('OWNER_PIN')
    return json.loads(raw)


def launch(configuration, configuration_sha256, activation, activation_sha256, root, socket_name,
           evidence_name, command, *, wall_seconds=60, stop_timeout=2, input_fd=0, output_fd=1):
    started = time.monotonic()
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (wall_seconds, stop_timeout)):
        raise ValueError('OWNER_LIMIT')
    if wall_seconds > 180 or wall_seconds <= 3 * stop_timeout + .75: raise ValueError('OWNER_LIMIT')
    root = private_root(root)
    if stat.S_IMODE(root.stat().st_mode) != 0o700: raise ValueError('OWNER_PRIVATE_ROOT')
    if any((path / '.git').exists() for path in (root, *root.parents)): raise ValueError('OWNER_PRIVATE_ROOT')
    if any(type(name) is not str or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name) for name in (socket_name, evidence_name)):
        raise ValueError('OWNER_PATH')
    config = pinned_json(configuration, configuration_sha256)
    route = pinned_json(activation, activation_sha256)
    evidence_fd = os.open(root/evidence_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    worker = endpoint = None; controls = []; previous = {}
    stopped = threading.Event(); counts = {'input_bytes': 0, 'output_bytes': 0}
    result = {'status': 'FAILED', 'reason': 'launch_failure', 'cleanup_confirmed': False, 'owner_thread_joined': False}
    deadline = started + wall_seconds
    try:
        endpoint = OwnerSocket(root/socket_name, deadline)
        worker = launch_owner_worker(command, config, root=root, activation=route,
                                     wall_seconds=deadline - time.monotonic() - .25, stop_timeout=stop_timeout)
        worker.deadline = min(worker.deadline, deadline - .1)
        result['worker_configuration_sha256'] = worker.config_sha256
        endpoint.start(worker.activation)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *args: stopped.set())
        read_fd, write_fd = os.pipe(); controls = [read_fd, write_fd]
        try:
            reason = relay(worker.process, read_fd, input_fd, output_fd, deadline - worker.cleanup - .25,
                           lambda: stopped.is_set() or endpoint.failed.is_set(), 262144, 16777216, counts)
        except RelayStop as error: reason = str(error)
        if endpoint.failed.is_set(): reason = 'owner_control'
        result['reason'] = reason
        endpoint.close(); result['owner_requests'] = endpoint.requests; endpoint = None
        result['owner_thread_joined'] = True
        if worker.process.stdin is not None: worker.process.stdin.close()
        receipt = worker.finish(abort=reason not in ('input_eof', 'child_exit'))
        result.update(worker=receipt, cleanup_confirmed=True)
        if reason in ('input_eof', 'child_exit') and receipt['returncode'] == 0 and time.monotonic() < deadline:
            result['status'] = 'PASSED'
    except Exception:
        result['reason'] = 'owner_failure'
    finally:
        if endpoint is not None:
            try:
                endpoint.close(); result['owner_thread_joined'] = True
                result['owner_requests'] = endpoint.requests
            except Exception: result['owner_thread_joined'] = False
        if worker is not None and not worker.closed:
            try: result['worker'] = worker.finish(abort=True); result['cleanup_confirmed'] = True
            except Exception: result['cleanup_confirmed'] = False
        result['cleanup_confirmed'] = result['cleanup_confirmed'] and result['owner_thread_joined']
        if time.monotonic() >= deadline: result['status'] = 'FAILED'
        for sig, handler in previous.items(): signal.signal(sig, handler)
        for fd in controls: os.close(fd)
        result.update(wall_seconds=time.monotonic()-started, **counts)
        try: os.write(evidence_fd, json.dumps(result, allow_nan=False).encode()+b'\n')
        finally: os.close(evidence_fd)
    return 0 if result['status'] == 'PASSED' and result['cleanup_confirmed'] else 2


def main():
    parser = argparse.ArgumentParser()
    for name in ('configuration', 'configuration-sha256', 'activation', 'activation-sha256', 'root', 'socket-name', 'evidence-name'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--wall-seconds', type=float, default=60)
    parser.add_argument('--stop-timeout', type=float, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    if args['command'][:1] == ['--']: args['command'] = args['command'][1:]
    try: return launch(**args)
    except Exception: return 2


if __name__ == '__main__': raise SystemExit(main())
