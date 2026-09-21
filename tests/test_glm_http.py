"""Exercise the actual owned HTTP child against bounded localhost fixtures."""
import json
import threading
import time
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from drift.serving.glm_http import GlmHttp
from drift.serving.glm_session import GlmSession


@pytest.fixture
def server():
    seen, disconnected = [], threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, body))
            self.send_response(200); self.end_headers()
            if self.path == '/tokenize':
                self.wfile.write(b'{"tokens":[1,2,3]}')
                return
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"private-fixture"},"finish_reason":null}]}\n\n')
            self.wfile.flush()
            if body.get('hold'):
                self.connection.settimeout(4)
                if self.connection.recv(1) == b'':
                    disconnected.set()
                return
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
            self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n')
            self.wfile.write(b'data: [DONE]\n\n')

    service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    yield f'http://127.0.0.1:{service.server_port}', seen, disconnected
    service.shutdown(); service.server_close(); thread.join(2)


def test_actual_child_tokenizes_and_streams_then_reaps(server):
    base, seen, _ = server
    transport = GlmHttp(base)
    backend = GlmSession(transport, model='fixture')
    backend.open({'limits': {'max_input_bytes': 4096, 'max_session_tokens': 64, 'deadline_ms': 10000},
                  'system_prompt': [], 'tools': []})
    backend.own_prompt({'text': 'own prompt'})
    events = list(backend.stream(8))
    assert events[-1]['payload']['usage'] == {'input_tokens': 3, 'output_tokens': 1}
    assert [path for path, _ in seen] == ['/tokenize', '/v1/chat/completions']
    assert transport.child is None


def test_cancel_terminates_owned_connection_and_child(server):
    base, _, disconnected = server
    transport = GlmHttp(base)
    events = transport.stream({'hold': True}, 4)
    assert next(events)['choices'][0]['delta']['content'] == 'private-fixture'
    child = transport.child
    transport.cancel()
    assert child.poll() is not None and transport.child is None
    assert disconnected.wait(2)
    events.close()
    assert child.stdout.closed


def test_absolute_stream_deadline_reaps_even_when_server_keeps_connection_open(server):
    transport = GlmHttp(server[0])
    started = time.monotonic()
    with pytest.raises((TimeoutError, ValueError)):
        list(transport.stream({'hold': True}, 1.2))
    assert time.monotonic() - started < 1.7 and transport.child is None


def test_cancel_before_request_prevents_late_dispatch(server):
    transport = GlmHttp(server[0])
    transport.cancel()
    with pytest.raises(ValueError):
        transport.count_tokens({'model': 'fixture', 'messages': []}, 3)
    assert server[1] == []


def test_stalled_child_input_cannot_escape_absolute_deadline(monkeypatch):
    actual = subprocess.Popen
    def blocked(command, **kwargs):
        return actual([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(60)'], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', blocked)
    transport, errors = GlmHttp(), []
    def request():
        try:
            transport.count_tokens({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'x' * 200000}]}, 1.2)
        except Exception as error:
            errors.append(type(error).__name__)
    thread = threading.Thread(target=request, daemon=True); thread.start()
    try:
        thread.join(1.8)
        assert not thread.is_alive(), 'input delivery escaped the request deadline'
        assert errors and transport.child is None
    finally:
        transport.cancel(); thread.join(2)


@pytest.mark.parametrize('base', ['https://example.com', 'http://127.0.0.1/path', 'http://key@127.0.0.1', 'http://127.0.0.1?secret=1'])
def test_http_transport_refuses_nonlocal_or_ambiguous_destination(base):
    with pytest.raises(ValueError):
        GlmHttp(base)
