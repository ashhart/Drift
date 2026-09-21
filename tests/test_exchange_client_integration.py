"""Exercise the shipped JS client against the Python coordinator over local sockets."""
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from drift.exchange.coordinator import ExchangeCoordinator
from tests.test_exchange_coordinator import make_session, talk


def test_js_client_reconnects_without_resetting_the_route():
    script = """
import assert from 'node:assert/strict';
import { exchangeRequest } from './scripts/omp/exchange_client.mjs';
for (let sequence = 0; sequence < 3; sequence++) {
  const reply = await exchangeRequest(process.argv[1], {op:'exchange',route:'r'}, {timeoutMs:1000});
  assert.equal(reply.published.sequence, sequence);
}
"""
    with tempfile.TemporaryDirectory(prefix='dx-', dir='/tmp') as directory:
        session = make_session()
        coordinator = ExchangeCoordinator(Path(directory) / 'x.sock', {'r': session}, time.monotonic() + 5)
        coordinator.start()
        try:
            result = subprocess.run([shutil.which('node') or 'node', '--input-type=module', '-e', script,
                                     str(coordinator.path)], capture_output=True, text=True, timeout=4)
            assert result.returncode == 0, result.stderr
            assert session.sequence == 3 and not coordinator.failed.is_set()
        finally:
            coordinator.close()


def test_reconnections_cannot_reset_the_request_budget():
    with tempfile.TemporaryDirectory(prefix='dx-', dir='/tmp') as directory:
        session = make_session()
        coordinator = ExchangeCoordinator(Path(directory) / 'x.sock', {'r': session}, time.monotonic() + 5, max_requests=1)
        coordinator.start()
        try:
            talk(coordinator.path, [{'op': 'exchange', 'route': 'r'}])
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(1)
                client.connect(str(coordinator.path))
                client.sendall(b'{"op":"exchange","route":"r"}\n')
                assert client.recv(4096) == b''
            assert coordinator.failed.wait(1) and session.sequence == 1
        finally:
            coordinator.close()


def test_real_js_abort_stops_the_coordinator_before_publication():
    script = """
import assert from 'node:assert/strict';
import { createInterface } from 'node:readline';
import { exchangeRequest } from './scripts/omp/exchange_client.mjs';
const input = createInterface({input:process.stdin});
const control = new AbortController();
input.once('line', () => control.abort());
await assert.rejects(exchangeRequest(process.argv[1], {op:'exchange',route:'r'},
  {timeoutMs:3000,signal:control.signal}), /EXCHANGE_CANCELLED/);
input.close();
"""
    with tempfile.TemporaryDirectory(prefix='dx-', dir='/tmp') as directory:
        session = make_session()
        original = session.source.snapshot
        entered, release = threading.Event(), threading.Event()

        def snapshot(copies):
            entered.set()
            assert release.wait(3)
            return original(copies)

        session.source.snapshot = snapshot
        coordinator = ExchangeCoordinator(Path(directory) / 'x.sock', {'r': session}, time.monotonic() + 5)
        coordinator.start()
        client = subprocess.Popen([shutil.which('node') or 'node', '--input-type=module', '-e', script,
                                   str(coordinator.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        try:
            assert entered.wait(2)
            _, error = client.communicate('abort\n', timeout=2)
            assert client.returncode == 0, error
            release.set()
            coordinator.thread.join(1)
            assert session.poisoned and session.sequence == 0 and session.link.delivered == []
        finally:
            release.set()
            if client.poll() is None:
                client.kill()
                client.communicate(timeout=2)
            coordinator.close()
