"""Verify cancellation against a local synthetic SSE server, without model inference."""
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
def sse_server():
    disconnected = threading.Event()
    complete = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/tokenize":
                body = b'{"tokens":[42]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"text":"synthetic chunk"}]}\n\n')
            if complete.is_set():
                self.wfile.write(b'data: [DONE]\n\n')
                self.close_connection = True
            self.wfile.flush()
            if not complete.is_set():
                self.connection.settimeout(5)
                if not self.connection.recv(1):
                    disconnected.set()
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", disconnected, complete
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def start_session(tmp_path, base, controlled):
    messages = tmp_path / "messages.json"
    messages.write_text('[{"role":"user","content":"synthetic input"}]')
    return subprocess.Popen([sys.executable, str(ROOT / "scripts/live/spark_live_session.py"),
        "--session", "synthetic", "--messages", str(messages), "--no-link", "--base", base]
        + (["--control-stdin"] if controlled else []),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "DRIFT_GLM_KEY": "synthetic-key", "PYTHONPATH": str(ROOT)})


def wait_for_text(proc):
    observed = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while b'"text"' not in observed:
            assert selector.select(5), "session did not forward its synthetic stream"
            chunk = os.read(proc.stdout.fileno(), 4096)
            assert chunk, f"session exited before streaming: {proc.stderr.read().decode()}"
            observed.extend(chunk)
    return observed


@pytest.mark.parametrize("command", [b'{"op":"abort"}\n', b'', b'{"op":"pause"}\n', b'x' * 1024])
def test_control_abort_or_eof_closes_http_stream_and_never_emits_done(tmp_path, sse_server, command):
    base, disconnected, _ = sse_server
    proc = start_session(tmp_path, base, True)
    try:
        observed = wait_for_text(proc)
        if command:
            proc.stdin.write(command)
            proc.stdin.flush()
        else:
            proc.stdin.close()
            proc.stdin = None
        stdout, _ = proc.communicate(timeout=5)
        observed.extend(stdout)
        events = [json.loads(row) for row in observed.splitlines()]
        explicit_or_eof = command in (b'{"op":"abort"}\n', b'')
        assert proc.returncode == (2 if explicit_or_eof else 3)
        assert sum(event.get("cancelled") is True for event in events) == int(explicit_or_eof)
        assert disconnected.wait(2), "the cancelled child still owns its HTTP stream"
        assert not any(json.loads(row).get("done") for row in observed.splitlines())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(2)


@pytest.mark.parametrize("controlled", [False, True])
def test_normal_completion_preserves_default_stdin_eof_compatibility(tmp_path, sse_server, controlled):
    base, _, complete = sse_server
    complete.set()
    proc = start_session(tmp_path, base, controlled)
    try:
        if not controlled:
            proc.stdin.close()
            proc.stdin = None
        proc.wait(timeout=5)
        stdout = proc.stdout.read()
        assert proc.returncode == 0, proc.stderr.read().decode()
        assert sum(bool(json.loads(row).get("done")) for row in stdout.splitlines()) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(2)
        if proc.stdin:
            proc.stdin.close()


def test_supervisor_signal_closes_http_stream(tmp_path, sse_server):
    base, disconnected, _ = sse_server
    proc = start_session(tmp_path, base, True)
    try:
        observed = wait_for_text(proc)
        proc.terminate()
        stdout, _ = proc.communicate(timeout=5)
        assert proc.returncode != 0
        assert disconnected.wait(2)
        assert not any(json.loads(row).get("done") for row in (observed + stdout).splitlines())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(2)


@pytest.mark.parametrize("ending", ["raise SystemExit(7)", "import time; time.sleep(30)", "import time; time.sleep(31)"])
def test_premature_done_is_withheld_until_child_exit_and_child_is_reaped(ending):
    source = "import json,os; print(json.dumps({'text':'synthetic','pid':os.getpid()}),flush=True); print('{\"done\":true}',flush=True); " + ending
    if "31" in ending:
        source = "import signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); " + source
    supervisor = "import json,sys; from drift.serving.live_session_control import supervise; supervise([sys.executable,'-c',sys.argv[1]],sys.stdin.fileno(),lambda event:print(json.dumps(event),flush=True),stop_timeout=0.2)"
    proc = subprocess.Popen([sys.executable, "-c", supervisor, source], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
    try:
        observed = wait_for_text(proc)
        if "sleep" in ending:
            proc.stdin.write(b'{"op":"abort"}\n')
            proc.stdin.flush()
        stdout, _ = proc.communicate(timeout=5)
        events = [json.loads(row) for row in (observed + stdout).splitlines()]
        assert proc.returncode != 0
        assert not any(event.get("done") for event in events)
        with pytest.raises(ProcessLookupError):
            os.kill(events[0]["pid"], 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(2)
