from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from drift.transcript.capture import request


def test_owned_http_request_and_redirect_refusal(monkeypatch):
    monkeypatch.setenv('DRIFT_GLM_KEY', 'synthetic-fixture-key')
    monkeypatch.setenv('http_proxy', 'http://proxy.invalid:9999')
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, body, self.headers.get('Authorization')))
            if body.get('redirect'):
                self.send_response(307)
                self.send_header('Location', 'http://never-contact.invalid/v1/completions')
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"usage":{"prompt_tokens":2,"completion_tokens":1}}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/v1/completions'
        assert request(url, {'prompt': [1, 2]}, 2)['usage']['prompt_tokens'] == 2
        with pytest.raises(ValueError, match='REDIRECT'):
            request(url, {'redirect': True}, 2)
        assert len(calls) == 2
        assert calls[0] == ('/v1/completions', {'prompt': [1, 2]}, 'Bearer synthetic-fixture-key')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('url', ['http://remote.invalid/v1/completions',
                               'http://127.0.0.1/v1/chat/completions',
                               'http://user:secret@127.0.0.1/v1/completions',
                               'http://127.0.0.1/v1/completions?extra=1'])
def test_only_owned_loopback_completion_endpoint_is_admitted(url):
    with pytest.raises(ValueError, match='LOOPBACK'):
        request(url, {}, 1)
