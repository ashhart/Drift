"""Bound loopback metrics reads even when a peer continuously trickles bytes."""
import http.server
import subprocess
import threading
import time

import pytest

from drift.serving.live_metric_fetch import MetricFetchError, fetch_metrics


@pytest.fixture
def server():
    servers = []
    def start(mode, body=b'vllm:test 1\n'):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.server.requests += 1
                if mode == 'stalled-header':
                    time.sleep(0.7)
                    return
                if mode == 'redirect':
                    self.send_response(302)
                    self.send_header('Location', '/private')
                    self.end_headers()
                    return
                self.send_response(500 if mode == 'error' else 200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    for value in body:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        if mode == 'trickle': time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        httpd.requests = 0
        thread = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()
        servers.append((httpd, thread))
        return f'http://127.0.0.1:{httpd.server_port}/metrics', httpd
    yield start
    for httpd, thread in servers:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize('mode', ['trickle', 'stalled-header'])
def test_trickling_body_cannot_extend_deadline_and_owned_child_is_reaped(server, monkeypatch, mode):
    url, _ = server(mode, b'01234567890123456789')
    children = []
    real_popen = subprocess.Popen
    def launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', launch)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='deadline'):
        fetch_metrics(url, timeout=0.3)
    assert time.monotonic() - started < 0.45
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed


def test_complete_metrics_return_without_logging_body(server, capfd):
    body = b'vllm:test{engine="0"} 1\n'
    url, _ = server('normal', body)
    assert fetch_metrics(url, timeout=2) == body.decode()
    assert capfd.readouterr() == ('', '')


@pytest.mark.parametrize('mode,body', [('normal', b'x' * 129), ('error', b'private-body'),
                                      ('redirect', b''), ('normal', b'\xff')])
def test_bad_or_oversized_response_has_only_fixed_error(server, capfd, mode, body):
    url, httpd = server(mode, body)
    with pytest.raises(MetricFetchError) as caught:
        fetch_metrics(url, timeout=2, max_bytes=128)
    assert str(caught.value) == 'metrics fetch failed'
    assert httpd.requests == 1
    assert capfd.readouterr() == ('', '')


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True])
def test_invalid_timeout_never_launches(timeout, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('unexpected child'))
    with pytest.raises(ValueError): fetch_metrics('http://127.0.0.1:8888/metrics', timeout=timeout)


@pytest.mark.parametrize('url', ['http://example.com/metrics', 'http://localhost/metrics',
    'http://127.0.0.1:8888/private', 'http://user:secret@127.0.0.1:8888/metrics'])
def test_only_explicit_loopback_metrics_endpoint_is_allowed(url, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('unexpected child'))
    with pytest.raises(ValueError): fetch_metrics(url, timeout=1)


def test_spawn_failure_does_not_expose_exception_payload(monkeypatch):
    def fail(*args, **kwargs): raise OSError('private-payload')
    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(MetricFetchError, match='^metrics fetch failed$'):
        fetch_metrics('http://127.0.0.1:8888/metrics', timeout=1)


def test_unconfirmed_cleanup_cannot_be_reported_as_normal_timeout(monkeypatch):
    import io
    class Child:
        stdout = io.BytesIO()
        def communicate(self, *, timeout): raise subprocess.TimeoutExpired('private-command', timeout)
        def poll(self): return None
        def kill(self): pass
        def wait(self, *, timeout):
            assert 0 <= timeout <= 1
            raise subprocess.TimeoutExpired('private-command', timeout)
    child = Child()
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: child)
    with pytest.raises(MetricFetchError, match='^metrics fetch cleanup unconfirmed$'):
        fetch_metrics('http://127.0.0.1:8888/metrics', timeout=1)
    assert child.stdout.closed
