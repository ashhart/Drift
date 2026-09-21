"""Exercise the real runner, owned client and observer against local HTTP fixtures."""
import json
import sys
import threading

import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from drift.serving.live_cancel_observation import CancellationObserver
from drift.serving.live_metrics import parse_metrics
from drift.serving.live_metric_fetch import fetch_metrics
from drift.serving.live_qualification_client import Limits, OwnedSession, claim_session
from scripts.qualify_live.cancel_driver import drive

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("service_key", ["synthetic-key", ""])
def test_real_client_runner_abort_returns_engine_to_idle(tmp_path, monkeypatch, service_key):
    active, disconnected = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_GET(self):
            count = int(active.is_set())
            values = {'num_requests_running': count, 'num_requests_waiting': 0, 'kv_cache_usage_perc': count / 10}
            body = '\n'.join(f'vllm:{name}{{engine="0",model_name="GLM-5.3-Flash-EXL3"}} {value}'
                             for name, value in values.items()).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            expected_auth = 'Bearer ' + service_key if service_key else None
            assert self.headers.get('Authorization') == expected_auth
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/tokenize':
                body = b'{"tokens":[42]}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            active.set()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"text":"synthetic-private-output"}]}\n\n')
            self.wfile.flush()
            self.connection.settimeout(5)
            try:
                if self.connection.recv(1) == b'':
                    disconnected.set()
            finally:
                active.clear()
                self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('DRIFT_GLM_KEY', service_key)
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    claim = claim_session(tmp_path / 'ledger')
    messages = claim.path / 'messages.json'
    messages.write_text('[{"role":"user","content":"synthetic"}]')
    base = f'http://127.0.0.1:{server.server_port}'
    limits = Limits(max_new=256, max_seconds=10)
    command = [sys.executable, str(ROOT / 'scripts/live/spark_live_session.py'),
               '--session', claim.name, '--messages', str(messages), '--base', base,
               '--max-new', '256', '--reserve', '8', '--control-stdin', '--no-link']
    client = OwnedSession(command, claim, limits)
    def sample(*, timeout):
        return parse_metrics(fetch_metrics(base + '/metrics', timeout=min(timeout, 2)))
    try:
        result = drive(CancellationObserver(timeout=10), client, sample, poll_seconds=0.02, max_seconds=10)
        assert result['status'] == 'PASSED', result
        assert disconnected.is_set() and not active.is_set()
        assert result['client']['cancelled'] and result['client']['child_terminated']
        assert result['client']['child_returncode'] == 2
        assert result['client']['cancelled_events'] == 1 and result['client']['done_events'] == 0
        assert 'synthetic-private-output' not in json.dumps(result)
        assert result['settled_samples'] == 3
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
