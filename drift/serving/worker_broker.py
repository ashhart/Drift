"""Launch an opt-in local activation channel beside the unchanged private worker stdio."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from drift.serving.worker_activation_client import ActivationClient
from drift.serving.worker_activation_files import private_root
from drift.serving.worker_broker_lifetime import BrokerWorker


ACTIVATION_FIELDS = {'session', 'source_worker', 'target_worker', 'memory_root', 'max_bytes', 'max_rows', 'max_total_rows'}


def _configuration(configuration, activation):
    if type(configuration) is not dict or 'activation' in configuration: raise ValueError('OWNER_CONFIGURATION')
    frozen = json.loads(json.dumps(configuration, allow_nan=False))
    if activation is None: return frozen, None
    if type(activation) is not dict or set(activation) != ACTIVATION_FIELDS: raise ValueError('OWNER_ACTIVATION')
    activation = json.loads(json.dumps(activation, allow_nan=False))
    for key in ('session', 'source_worker', 'target_worker'):
        if type(activation[key]) is not str or not 0 < len(activation[key]) <= 96: raise ValueError('OWNER_IDENTITY')
    if activation['source_worker'] == activation['target_worker'] or activation['target_worker'] != frozen.get('worker'): raise ValueError('OWNER_IDENTITY')
    private_root(activation['memory_root'])
    for key, maximum in (('max_bytes', 128 * 1024 * 1024), ('max_rows', 4096), ('max_total_rows', 1_000_000)):
        if type(activation[key]) is not int or not 1 <= activation[key] <= maximum: raise ValueError('OWNER_LIMIT')
    return frozen, activation


def launch_owner_worker(command, configuration, *, root, activation=None, wall_seconds, stop_timeout=2, cwd=None, env=None):
    if type(command) not in (list, tuple) or not command or any(type(value) is not str for value in command) or list(command).count('{worker_config}') != 1:
        raise ValueError('OWNER_COMMAND')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (wall_seconds, stop_timeout)) or wall_seconds <= 3 * stop_timeout + .25:
        raise ValueError('OWNER_LIMIT')
    config, activation = _configuration(configuration, activation)
    directory = Path(tempfile.mkdtemp(prefix='worker-owner-', dir=private_root(root)))
    config_path = directory / 'worker.json'
    fds, pair, process = [], None, None
    deadline = time.monotonic() + wall_seconds
    try:
        control_r, control_w = os.pipe(); fds.extend((control_r, control_w))
        evidence_r, evidence_w = os.pipe(); fds.extend((evidence_r, evidence_w))
        inherited = [control_r, evidence_w]
        extra = []
        if activation is not None:
            pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            config['activation'] = {**activation, 'fd': pair[1].fileno()}
            inherited.append(pair[1].fileno()); extra = ['--activation-fd', str(pair[1].fileno())]
        raw = json.dumps(config, sort_keys=True, allow_nan=False).encode()
        if len(raw) > 1048576: raise ValueError('OWNER_CONFIGURATION_LIMIT')
        fd = os.open(config_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as handle: handle.write(raw)
        config_path.chmod(0o400)
        concrete = [str(config_path) if value == '{worker_config}' else value for value in command]
        supervisor = [sys.executable, '-m', 'drift.serving.worker_supervisor', '--control-fd', str(control_r), '--evidence-fd', str(evidence_w), '--input-fd', '0', '--output-fd', '1', '--wall-seconds', str(wall_seconds), '--stop-timeout', str(stop_timeout), *extra, '--', *concrete]
        process = subprocess.Popen(supervisor, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, pass_fds=tuple(inherited), cwd=cwd, env=env)
        os.close(control_r); fds.remove(control_r)
        os.close(evidence_w); fds.remove(evidence_w)
        if pair is not None: pair[1].close()
        client = ActivationClient(pair[0], activation, deadline) if pair is not None else None
        return BrokerWorker(process, control_w, evidence_r, directory, config_path, hashlib.sha256(raw).hexdigest(), client, deadline, 3 * stop_timeout + .5)
    except Exception:
        for fd in fds: os.close(fd)
        if pair is not None:
            for endpoint in pair: endpoint.close()
        shutil.rmtree(directory)
        if process is not None:
            try: process.communicate(timeout=3 * stop_timeout + .5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.communicate(timeout=3 * stop_timeout + .5)
        raise
