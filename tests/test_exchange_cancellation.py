"""Refuse work after the requesting boundary disappears or its budget expires."""
import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from drift.exchange.coordinator import ExchangeCoordinator
from tests.test_exchange_coordinator import make_session


@pytest.mark.parametrize('phase', ['snapshot', 'prepare', 'commit'])
@pytest.mark.parametrize('reason', ['disconnect', 'deadline', 'shutdown'])
def test_cancelled_boundary_cannot_advance_to_the_next_mutating_phase(phase, reason):
    session = make_session()
    session.link.queue = [dict(rows=1, start=0, stop=1)]
    owner = session.source if phase == 'snapshot' else session.sink
    original = getattr(owner, phase)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def pause(*args):
        entered.set()
        assert release.wait(2)
        result = original(*args)
        returned.set()
        return result

    setattr(owner, phase, pause)
    with tempfile.TemporaryDirectory(prefix='dx-', dir='/tmp') as directory:
        coordinator = ExchangeCoordinator(Path(directory) / 'x.sock', {'r': session}, time.monotonic() + 5)
        coordinator.start()
        client = socket.socket(socket.AF_UNIX)
        client.settimeout(2)
        client.connect(str(coordinator.path))
        try:
            client.sendall(b'{"op":"exchange","route":"r"}\n')
            assert entered.wait(1)
            if reason == 'disconnect':
                client.shutdown(socket.SHUT_RDWR)
                client.close()
            elif reason == 'deadline':
                coordinator.deadline = time.monotonic() - 1
            else:
                coordinator.stopped.set()
            release.set()
            assert returned.wait(1)
            if reason != 'disconnect':
                reply = client.recv(4096)
                assert not reply or json.loads(reply)['op'] == 'error'
            coordinator.thread.join(1)
        finally:
            release.set()
            client.close()
            coordinator.close()
    assert session.poisoned
    if phase == 'snapshot':
        assert session.link.delivered == [] and session.sequence == 0
    if phase != 'commit':
        assert session.sink.applied == []
    assert session.link.acked == []
