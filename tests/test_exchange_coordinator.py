"""Drive the coordinator over a real Unix socket with fake runtimes: no model, host or MCDMA."""
import json
import socket
import tempfile
import time
from pathlib import Path

import pytest

from drift.exchange.coordinator import ExchangeCoordinator, dispatch
from drift.exchange.session import ExchangeSession
from tests.test_exchange_session import Link, Sink, Source


def make_session(mode='drift'):
    return ExchangeSession(session='live-1', source_worker='qwen', target_worker='glm', mode=mode,
                           ranks=('spark-a.invalid', 'spark-b.invalid'), source=Source(), link=Link(), sink=Sink(), copies=12)


def test_exchange_publishes_and_drains_in_one_bounded_request():
    session = make_session()
    session.link.queue = [dict(rows=2, start=0, stop=2)]
    reply = dispatch({'qwen-to-glm': session}, {'op': 'exchange', 'route': 'qwen-to-glm'})
    assert reply['op'] == 'exchanged' and reply['published']['rows'] == 12
    assert reply['published']['applied_ranks'] == ['spark-a.invalid', 'spark-b.invalid']
    assert [item['rows'] for item in reply['applied']] == [2] and reply['foreign_rows'] == 2


def test_status_reports_cursors_without_advancing_them():
    session = make_session()
    before = dispatch({'r': session}, {'op': 'status', 'route': 'r'})
    assert before == dict(op='status', route='r', mode='drift', sequence=0, foreign_rows=0, poisoned=False)
    assert session.sequence == 0


@pytest.mark.parametrize('frame', [
    {'op': 'exchange'}, {'op': 'exchange', 'route': 'r', 'extra': 1}, {'op': 'drop', 'route': 'r'},
    {'op': 'exchange', 'route': 'missing'}, {'op': 'exchange', 'route': 7}, ['exchange'],
])
def test_a_malformed_or_unknown_frame_is_refused(frame):
    with pytest.raises(ValueError):
        dispatch({'r': make_session()}, frame)


def test_the_reply_carries_no_tuples_so_it_survives_json():
    session = make_session()
    reply = dispatch({'r': session}, {'op': 'exchange', 'route': 'r'})
    assert json.loads(json.dumps(reply, allow_nan=False))['published']['receipts'] == ['b' * 64, 'b' * 64]


@pytest.fixture
def short_dir():
    """A Unix socket path has to fit sun_path, which pytest's tmp_path does not on macOS."""
    with tempfile.TemporaryDirectory(prefix='/tmp/dx-'[:8]) as directory:
        yield Path(directory)


def test_a_path_that_cannot_fit_sun_path_is_refused_before_binding(tmp_path):
    long_path = tmp_path / ('n' * 120 + '.sock')
    with pytest.raises(ValueError):
        ExchangeCoordinator(long_path, {'r': make_session()}, time.monotonic() + 5)


def talk(path, frames):
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(5)
    client.connect(str(path))
    replies = []
    for frame in frames:
        client.sendall(json.dumps(frame).encode() + b'\n')
        raw = bytearray()
        while b'\n' not in raw:
            raw.extend(client.recv(4096))
        replies.append(json.loads(raw))
    client.close()
    return replies


def test_a_real_peer_gets_one_reply_per_request_over_the_socket(short_dir):
    session = make_session()
    session.link.queue = [dict(rows=1, start=0, stop=1)]
    coordinator = ExchangeCoordinator(short_dir / 'x.sock', {'r': session}, time.monotonic() + 10)
    coordinator.start()
    try:
        replies = talk(coordinator.path, [{'op': 'exchange', 'route': 'r'}, {'op': 'status', 'route': 'r'}])
    finally:
        coordinator.close()
    assert replies[0]['op'] == 'exchanged' and replies[0]['published']['sequence'] == 0
    assert replies[1] == dict(op='status', route='r', mode='drift', sequence=1, foreign_rows=1, poisoned=False)
    assert not coordinator.path.exists()


def test_the_socket_is_private_to_its_owner(short_dir):
    coordinator = ExchangeCoordinator(short_dir / 'x.sock', {'r': make_session()}, time.monotonic() + 5)
    try:
        assert (coordinator.path.stat().st_mode & 0o777) == 0o600
    finally:
        coordinator.close()


def test_a_poisoned_route_reports_an_error_frame_and_stays_poisoned(short_dir):
    session = make_session()
    session.link.confirm_applied = lambda delivered, rows: dict(ranks=('spark-a.invalid',), receipts=('b' * 64,))
    coordinator = ExchangeCoordinator(short_dir / 'x.sock', {'r': session}, time.monotonic() + 10)
    coordinator.start()
    try:
        replies = talk(coordinator.path, [{'op': 'exchange', 'route': 'r'}, {'op': 'status', 'route': 'r'}])
    finally:
        coordinator.close()
    assert replies[0]['op'] == 'error' and replies[0]['code'].startswith('EXCHANGE_')
    assert replies[1]['poisoned'] is True


def test_an_oversized_frame_never_reaches_a_session(short_dir):
    session = make_session()
    coordinator = ExchangeCoordinator(short_dir / 'x.sock', {'r': session}, time.monotonic() + 10)
    coordinator.start()
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(5)
    client.connect(str(coordinator.path))
    try:
        client.sendall(b'{"op":"exchange","route":"' + b'r' * 5000 + b'"}\n')
        time.sleep(.3)
    finally:
        client.close()
        coordinator.close()
    assert session.sequence == 0 and coordinator.failed.is_set()
