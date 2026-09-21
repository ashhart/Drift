import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import time

import pytest
from test_worker_broker import fixture
from test_worker_activation import append_frame


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="own-", dir="/tmp") as name:
        yield Path(name).resolve()


def launch(tmp_path, *, wall=4):
    root, memory, command, activation = fixture(tmp_path)
    script = Path(command[1])
    code = script.read_text().replace("'own_slots':[]", "'own_slots':[0,1]")
    code = code.replace(" if command['op']=='append':", " if command['op']=='tap':\n  import numpy as np\n  np.savez(command['out'],k0=np.zeros((2,1,2),dtype=np.float16),v0=np.zeros((2,1,2),dtype=np.float16))\n  return {'tapped':2,'next_first':2}\n if command['op']=='append':")
    script.write_text(code)
    config = root / 'base.json'; config.write_text(json.dumps({'worker': 'qwen'}))
    route = root / 'route.json'; route.write_text(json.dumps(activation))
    import hashlib
    arguments = [sys.executable, '-m', 'drift.serving.worker_owner_stdio', '--configuration', str(config),
                 '--configuration-sha256', hashlib.sha256(config.read_bytes()).hexdigest(), '--activation', str(route),
                 '--activation-sha256', hashlib.sha256(route.read_bytes()).hexdigest(), '--root', str(root),
                 '--socket-name', 'owner.sock', '--evidence-name', 'termination.json', '--wall-seconds', str(wall),
                 '--stop-timeout', '.1', '--', *command]
    child = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return child, root, memory


def line(child):
    assert select.select([child.stdout], [], [], 3)[0]
    raw = child.stdout.readline()
    assert raw, child.stderr.read().decode()
    return json.loads(raw)


def connect(root):
    peer = socket.socket(socket.AF_UNIX); peer.settimeout(2)
    peer.connect(str(root / 'owner.sock'))
    return peer


def reply(peer, frame):
    peer.sendall(json.dumps(frame).encode() + b'\n')
    return json.loads(peer.recv(4096))


def receipt(child, root):
    child.wait(timeout=5)
    value = json.loads((root / 'termination.json').read_text())
    assert 'worker' in value, value
    assert value['worker']['child_reaped'] and not value['worker']['process_group_alive']
    assert not (root / 'owner.sock').exists()
    return value


def test_owner_socket_is_separate_from_own_stdio_and_eof_reaps(tmp_path):
    child, root, memory = launch(tmp_path)
    try:
        assert line(child) == {'ready': True, 'activation': True}
        assert root.stat().st_mode & 0o777 == 0o700
        assert (root / 'owner.sock').stat().st_mode & 0o777 == 0o600
        with connect(root) as peer:
            frame = append_frame(memory)
            result = reply(peer, {key: frame[key] for key in ('op', 'path', 'sha256', 'rows')})
            assert result['op'] == 'appended' and result['rows'] == 2
            tapped = reply(peer, {'op': 'tap', 'path': 'out.npz', 'first': 0, 'max_rows': 4})
            assert tapped['op'] == 'tapped' and tapped['rows'] == 2
            assert (tapped['source_worker'], tapped['target_worker']) == ('qwen', 'glm')
            assert (result['source_worker'], result['target_worker']) == ('glm', 'qwen')
            assert (memory/'out.npz').exists()
            child.stdin.write(b'{"op":"observe"}\n'); child.stdin.flush()
            assert line(child) == {'foreign_total': 2}
            child.stdin.close()
            evidence = receipt(child, root)
            assert evidence['reason'] == 'input_eof' and evidence['status'] == 'PASSED'
            assert evidence['owner_requests'] == 2 and evidence['owner_thread_joined']
            assert len(evidence['worker_configuration_sha256']) == 64
    finally:
        if child.poll() is None: child.kill(); child.wait()


@pytest.mark.parametrize('operation', ['malformed', 'duplicate', 'oversized', 'eof', 'abort'])
def test_owner_control_failure_aborts_and_reaps(tmp_path, operation):
    child, root, memory = launch(tmp_path)
    try:
        assert line(child)['ready']
        peer = connect(root)
        if operation == 'malformed': peer.sendall(b'{"op":"append","secret":"NEVER_EVIDENCE"}\n')
        elif operation == 'duplicate': peer.sendall(b'{"op":"tap","op":"abort"}\n')
        elif operation == 'oversized': peer.sendall(b'x'*4097)
        elif operation == 'abort': peer.sendall(b'{"op":"abort"}\n')
        else: peer.close()
        value = receipt(child, root)
        assert value['status'] == 'FAILED' and value['reason'] == 'owner_control'
        assert 'NEVER_EVIDENCE' not in json.dumps(value)
        peer.close()
    finally:
        if child.poll() is None: child.kill(); child.wait()


def test_deadline_is_independent_of_open_stdio_and_owner_socket(tmp_path):
    started = time.monotonic()
    child, root, memory = launch(tmp_path, wall=1.3)
    try:
        assert line(child)['ready']
        with connect(root):
            value = receipt(child, root)
        assert value['status'] == 'FAILED' and time.monotonic() - started < 2.5
    finally:
        if child.poll() is None: child.kill(); child.wait()


def test_malformed_own_input_terminates_without_exporting_private_exception(tmp_path):
    child, root, memory = launch(tmp_path)
    try:
        assert line(child)['ready']
        with connect(root):
            child.stdin.write(b'NEVER_EVIDENCE\n'); child.stdin.flush()
            value = receipt(child, root)
            assert value['status'] == 'FAILED'
            assert 'NEVER_EVIDENCE' not in json.dumps(value)
    finally:
        if child.poll() is None: child.kill(); child.wait()
