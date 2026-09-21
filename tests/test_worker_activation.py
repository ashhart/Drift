import hashlib
import threading
import time
import numpy as np
import pytest
from drift.serving.worker_activation import ActivationControl, NativeGate


def control(tmp_path):
    root = tmp_path / 'memory'; root.mkdir(mode=0o700)
    state = {'cache': object(), 'own_slots': [0, 1], 'poisoned': False}
    events = []
    def handle(command):
        events.append(command['op'])
        if command['op'] == 'append':
            return {'appended': 2, 'foreign_total': 2, 'cache_slots': 4}
        if command['op'] == 'tap':
            np.savez(command['out'], k0=np.ones((2, 1, 2), dtype=np.float16), v0=np.ones((2, 1, 2), dtype=np.float16))
            return {'tapped': 2, 'next_first': 2}
    gate = NativeGate(handle, state, lambda: events.append('settled'), timeout=1)
    config = {'session': 'session-one', 'source_worker': 'glm', 'target_worker': 'qwen', 'memory_root': str(root), 'max_bytes': 65536, 'max_rows': 4, 'max_total_rows': 4}
    return ActivationControl(config, gate, {'k0': (1, 2), 'v0': (1, 2)}), root, state, events


def append_frame(root):
    path = root / 'incoming.npz'
    np.savez(path, k0=np.ones((2, 1, 2), dtype=np.float16), v0=np.ones((2, 1, 2), dtype=np.float16))
    return {'v': 1, 'session': 'session-one', 'seq': 1, 'op': 'append', 'path': 'incoming.npz', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': 2}


def test_append_uses_hashed_snapshot_and_acknowledges_only_after_settle(tmp_path):
    channel, root, state, events = control(tmp_path)
    reply = channel.apply(append_frame(root))
    assert reply['op'] == 'appended' and reply['rows'] == 2
    assert (reply['source_worker'], reply['target_worker']) == ('glm', 'qwen')
    assert events == ['append', 'settled']
    assert not state['poisoned']
    with pytest.raises(RuntimeError): channel.apply(append_frame(root))
    assert state['poisoned']


def test_hash_or_row_mismatch_never_reaches_native_handler(tmp_path):
    channel, root, state, events = control(tmp_path)
    frame = append_frame(root); frame['rows'] = 3
    with pytest.raises(RuntimeError): channel.apply(frame)
    assert events == [] and state['poisoned']


def test_tap_is_bounded_exclusive_and_preserves_cursor(tmp_path):
    channel, root, state, events = control(tmp_path)
    frame = {'v': 1, 'session': 'session-one', 'seq': 1, 'op': 'tap', 'path': 'out.npz', 'first': 0, 'max_rows': 2}
    reply = channel.apply(frame)
    assert reply['rows'] == 2 and reply['next_first'] == 2
    assert (reply['source_worker'], reply['target_worker']) == ('qwen', 'glm')
    assert reply['sha256'] == hashlib.sha256((root / 'out.npz').read_bytes()).hexdigest()
    assert events == ['tap', 'settled']
    frame['seq'] = 2
    with pytest.raises(RuntimeError): channel.apply(frame)


def test_path_escape_and_extra_text_fail_closed(tmp_path):
    channel, root, state, events = control(tmp_path)
    frame = append_frame(root); frame['path'] = '../out.npz'
    with pytest.raises(RuntimeError): channel.apply(frame)
    assert not events


def test_native_calls_do_not_overlap_and_failure_poisons_waiters():
    state = {'cache': object()}; entered = threading.Event(); release = threading.Event(); seen = []
    def handle(frame):
        seen.append(frame['op'])
        if frame['op'] == 'generate_own': entered.set(); release.wait(1)
        return {}
    gate = NativeGate(handle, state, lambda: None, timeout=1)
    one = threading.Thread(target=lambda: gate({'op': 'generate_own'})); one.start(); assert entered.wait(1)
    two = threading.Thread(target=lambda: gate({'op': 'append'})); two.start(); time.sleep(0.02)
    assert seen == ['generate_own']
    release.set(); one.join(1); two.join(1)
    assert seen == ['generate_own', 'append']


def test_inherited_activation_socket_is_separate_and_eof_poisons(tmp_path):
    import json
    import os
    import socket
    from drift.serving.worker_activation_server import ActivationServer
    channel, root, state, events = control(tmp_path)
    parent, child = socket.socketpair()
    failed = threading.Event()
    config = {'fd': os.dup(child.fileno()), 'session': 'session-one', 'source_worker': 'glm', 'target_worker': 'qwen', 'memory_root': str(root), 'max_bytes': 65536, 'max_rows': 4, 'max_total_rows': 4}
    server = ActivationServer(config, channel.gate, channel.layouts, failed.set)
    server.start(); child.close(); parent.settimeout(1)
    parent.sendall(json.dumps(append_frame(root)).encode() + b'\n')
    reply = json.loads(parent.recv(4096))
    assert reply['op'] == 'appended' and events[-1] == 'settled'
    parent.close()
    assert failed.wait(1) and state['poisoned']
    server.close()


def test_native_failure_poison_survives_handler_clearing_its_state_flag():
    state = {}; gate = None
    def handle(frame):
        gate.poison(); state['poisoned'] = False
        return {}
    gate = NativeGate(handle, state, lambda: None)
    with pytest.raises(RuntimeError): gate({'op': 'append'})
    assert state['poisoned'] and gate.failed.is_set()


def test_oversize_tap_fails_before_native_write(tmp_path):
    channel, root, state, events = control(tmp_path)
    channel.max_bytes = 5
    frame = {'v': 1, 'session': 'session-one', 'seq': 1, 'op': 'tap', 'path': 'out.npz', 'first': 0, 'max_rows': 2}
    with pytest.raises(RuntimeError): channel.apply(frame)
    assert events == [] and not (root / 'out.npz').exists()


def test_fifo_publication_cannot_block_activation_reader(tmp_path):
    import os
    channel, root, state, events = control(tmp_path)
    os.mkfifo(root / 'pipe.npz')
    frame = {'v': 1, 'session': 'session-one', 'seq': 1, 'op': 'append', 'path': 'pipe.npz', 'sha256': '0' * 64, 'rows': 2}
    before = time.monotonic()
    with pytest.raises(RuntimeError): channel.apply(frame)
    assert time.monotonic() - before < 0.5 and not events
