import json
import socket
import threading
import time
import pytest
from drift.serving.worker_activation_client import ActivationClient


def client():
    owned, peer = socket.socketpair()
    config = {'session': 's', 'source_worker': 'glm', 'target_worker': 'qwen', 'max_rows': 4, 'max_total_rows': 8}
    return ActivationClient(owned, config, time.monotonic() + .5), peer


@pytest.mark.parametrize('mutation', [{'seq': True}, {'unexpected': 'task text'}, {'foreign_total': 7}])
def test_invalid_receipt_poison_closes_controller_endpoint(mutation):
    owner, peer = client()
    def response():
        peer.recv(4096)
        value = {'v': 1, 'session': 's', 'seq': 1, 'source_worker': 'glm', 'target_worker': 'qwen', 'op': 'appended', 'rows': 2, 'sha256': 'a' * 64, 'foreign_total': 2, **mutation}
        peer.sendall(json.dumps(value).encode() + b'\n')
    thread = threading.Thread(target=response); thread.start()
    try:
        with pytest.raises(RuntimeError, match='ACTIVATION_CHANNEL_FAILED'): owner.append('memory.npz', 'a' * 64, 2)
        assert owner.socket.fileno() == -1
    finally:
        thread.join(); owner.close(); peer.close()


def test_no_text_frames_or_paths_and_absolute_deadline():
    owner, peer = client()
    try:
        with pytest.raises(ValueError): owner.append('../task.txt', 'a' * 64, 2)
        with pytest.raises(ValueError): owner.append('memory.npz', 'task text', 2)
        owner.deadline = time.monotonic() - 1
        with pytest.raises(RuntimeError): owner.tap('memory.npz', 0, 2)
        assert owner.socket.fileno() == -1
    finally:
        owner.close(); peer.close()


@pytest.mark.parametrize('operation', ['append', 'tap'])
@pytest.mark.parametrize('mutation', [None, 'swapped', 'source', 'target'])
def test_receipt_route_is_bound_to_operation(operation, mutation):
    owner, peer = client()
    source, target = ('glm', 'qwen') if operation == 'append' else ('qwen', 'glm')
    if mutation == 'swapped': source, target = target, source
    if mutation == 'source': source = 'forged'
    if mutation == 'target': target = 'forged'
    def response():
        peer.recv(4096)
        result = {'foreign_total': 2} if operation == 'append' else {'next_first': 2}
        value = {'v': 1, 'session': 's', 'seq': 1, 'source_worker': source, 'target_worker': target, 'op': 'appended' if operation == 'append' else 'tapped', 'rows': 2, 'sha256': 'a' * 64, **result}
        peer.sendall(json.dumps(value).encode() + b'\n')
    thread = threading.Thread(target=response); thread.start()
    try:
        invoke = lambda: owner.append('memory.npz', 'a' * 64, 2) if operation == 'append' else owner.tap('memory.npz', 0, 2)
        if mutation is None:
            assert invoke()['rows'] == 2
        else:
            with pytest.raises(RuntimeError, match='ACTIVATION_CHANNEL_FAILED'): invoke()
            assert owner.socket.fileno() == -1
    finally:
        thread.join(); owner.close(); peer.close()
