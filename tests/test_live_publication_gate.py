"""Exercise a gated real request runner against a synthetic HTTP service."""
import json
import os
import selectors
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gated_runner(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                payload = b'{"tokens":[42]}'
            else:
                requests.append(body)
                payload = b'data: {"choices":[{"text":"synthetic"}]}\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = tmp_path / 'runner.py'
    source = (ROOT / 'scripts/live/spark_live_session.py').read_text()
    assert source.count('Path("/dev/shm/glm53-handoff")') == 1
    script.write_text(source.replace('Path("/dev/shm/glm53-handoff")', f'Path({str(tmp_path / "cache")!r})'))
    messages = tmp_path / 'messages.json'
    messages.write_text('[{"role":"user","content":"synthetic input"}]')
    children = []
    def start(*extra, no_tap=True):
        command = [sys.executable, str(script), '--session', 'one', '--messages', str(messages),
                   '--base', f'http://127.0.0.1:{server.server_port}', '--control-stdin',
                   '--wait-publication', '--ready-timeout', '0.7', *extra]
        if no_tap: command.append('--no-tap')
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, 'DRIFT_GLM_KEY': '', 'PYTHONPATH': str(ROOT)})
        children.append(proc)
        return proc
    yield start, requests, tmp_path / 'cache'
    for proc in children:
        if proc.poll() is None: proc.kill()
        proc.wait(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream: stream.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def ready(proc):
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        assert selector.select(3)
        line = proc.stdout.readline()
    assert line, proc.stderr.read().decode()
    assert json.loads(line) == {'publication_ready': True}


def test_no_completion_post_before_release_and_no_tap_is_explicit(gated_runner):
    start, requests, cache = gated_runner
    proc = start()
    ready(proc)
    assert requests == []
    assert (cache / 'tp-live-in/one').is_dir()
    proc.stdin.write(b'{"op":"release"}\n')
    proc.stdin.flush()
    proc.wait(timeout=3)
    events = [json.loads(line) for line in proc.stdout.read().splitlines()]
    assert proc.returncode == 0 and sum(event.get('done') is True for event in events) == 1
    assert len(requests) == 1
    assert requests[0]['kv_transfer_params']['drift_tap'] is False


@pytest.mark.parametrize('control', [b'{"op":"abort"}\n', b'', None])
def test_abort_eof_and_deadline_while_waiting_never_post_completion(gated_runner, control):
    start, requests, _ = gated_runner
    proc = start()
    ready(proc)
    if control == b'':
        proc.stdin.close()
        proc.stdin = None
    elif control is not None:
        proc.stdin.write(control)
        proc.stdin.flush()
    proc.wait(timeout=3)
    events = [json.loads(line) for line in proc.stdout.read().splitlines()]
    assert proc.returncode in (2, 3) and requests == []
    assert not any(event.get('done') for event in events)


def test_reused_inbox_is_preserved_and_rejected(gated_runner):
    start, requests, cache = gated_runner
    folder = cache / 'tp-live-in/one'
    folder.mkdir(parents=True)
    marker = folder / 'evidence'
    marker.write_bytes(b'preserve')
    proc = start()
    proc.wait(timeout=3)
    assert proc.returncode != 0 and requests == [] and marker.read_bytes() == b'preserve'


def test_release_before_ready_is_rejected_without_completion(gated_runner):
    start, requests, _ = gated_runner
    proc = start()
    proc.stdin.write(b'{"op":"release"}\n')
    proc.stdin.flush()
    proc.wait(timeout=3)
    assert proc.returncode == 3 and requests == []


def test_supervisor_release_is_one_shot_after_readiness(tmp_path):
    child = "import json,os,sys,time; print('{\"publication_ready\":true}',flush=True); sys.stdin.readline(); print(json.dumps({'started':True,'pid':os.getpid()}),flush=True); time.sleep(20)"
    driver = "import json,sys; from drift.serving.live_session_control import supervise; supervise([sys.executable,'-c',sys.argv[1]],sys.stdin.fileno(),lambda event:print(json.dumps(event),flush=True),allow_release=True,stop_timeout=.2)"
    proc = subprocess.Popen([sys.executable, '-c', driver, child], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready(proc)
        proc.stdin.write(b'{"op":"release"}\n')
        proc.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            assert selector.select(3)
            event = json.loads(proc.stdout.readline())
        assert event['started'] is True
        proc.stdin.write(b'{"op":"release"}\n')
        proc.stdin.flush()
        proc.wait(timeout=3)
        assert proc.returncode != 0 and b'"done"' not in proc.stdout.read()
        with pytest.raises(ProcessLookupError): os.kill(event['pid'], 0)
    finally:
        if proc.poll() is None: proc.kill()
        proc.wait(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr): stream.close()


def test_linked_default_keeps_existing_tap_behavior(gated_runner):
    start, requests, _ = gated_runner
    proc = start(no_tap=False)
    ready(proc)
    proc.stdin.write(b'{"op":"release"}\n')
    proc.stdin.flush()
    proc.wait(timeout=3)
    assert proc.returncode == 0 and 'drift_tap' not in requests[0]['kv_transfer_params']
