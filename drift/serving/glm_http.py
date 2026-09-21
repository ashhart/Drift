"""Bound and reap a dedicated process for each native GLM HTTP operation."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from drift.serving.glm_http_input import write_input


class GlmHttp:
    def __init__(self, base='http://127.0.0.1:8888'):
        url = urlsplit(base)
        if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password or url.path or url.query or url.fragment:
            raise ValueError('only the owned loopback server is allowed')
        self.base, self.child, self.lock = base, None, threading.Lock()
        self.closed = False

    def _script(self, path):
        return Path(__file__).with_name('glm_http_child.py')

    def _spawn(self, path, body, timeout, deadline):
        if not isinstance(timeout, (int, float)) or not 1 < timeout <= 3600:
            raise ValueError('HTTP time budget unavailable')
        payload = json.dumps({'url': self.base + path, 'body': body, 'timeout': timeout - 0.5}, allow_nan=False).encode() + b'\n'
        if len(payload) > 1048576:
            raise ValueError('HTTP input bound')
        with self.lock:
            if self.child is not None or self.closed:
                raise ValueError('one owned HTTP request at a time')
            child = subprocess.Popen([sys.executable, '-I', '-S', str(self._script(path))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env={'DRIFT_GLM_KEY': os.environ.get('DRIFT_GLM_KEY', '')})
            self.child = child
        try:
            write_input(child, payload, deadline)
        except Exception:
            self._stop(child)
            child.stdin.close(); child.stdout.close()
            raise
        return child

    def _frames(self, path, body, timeout):
        deadline = time.monotonic() + timeout - 0.5
        child = self._spawn(path, body, timeout, deadline)
        buffer, total = bytearray(), 0
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('owned HTTP request deadline')
                    if not selector.select(min(0.1, remaining)):
                        continue
                    chunk = os.read(child.stdout.fileno(), 4096)
                    if not chunk:
                        if buffer or child.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
                            raise ValueError('owned HTTP request did not finish cleanly')
                        return
                    buffer.extend(chunk); total += len(chunk)
                    if total > 4194304:
                        raise ValueError('HTTP output bound')
                    while b'\n' in buffer:
                        raw, _, rest = buffer.partition(b'\n'); buffer = bytearray(rest)
                        if len(raw) > 131072:
                            raise ValueError('HTTP frame bound')
                        event = json.loads(raw)
                        if type(event) is not dict or 'transport_error' in event:
                            raise ValueError('owned HTTP operation failed')
                        yield event
                    if len(buffer) > 131072:
                        raise ValueError('HTTP frame bound')
        finally:
            self._stop(child)
            child.stdout.close()

    def count_tokens(self, body, timeout):
        keys = ('model', 'messages', 'tools', 'chat_template', 'chat_template_kwargs')
        request = {key: body[key] for key in keys if key in body}
        request['add_generation_prompt'] = True
        frames = list(self._frames('/tokenize', request, timeout))
        if len(frames) != 1 or set(frames[0]) != {'tokens'} or type(frames[0]['tokens']) is not int:
            raise ValueError('invalid token count')
        return frames[0]['tokens']

    def stream(self, body, timeout):
        done = False
        for frame in self._frames('/v1/chat/completions', body, timeout):
            if done:
                raise ValueError('data after stream end')
            if frame == {'done': True}:
                done = True
            elif set(frame) == {'event'}:
                yield frame['event']
            else:
                raise ValueError('invalid stream envelope')
        if not done:
            raise ValueError('missing stream end')

    def _stop(self, child):
        with self.lock:
            if self.child is not child:
                return
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=0.2)
            self.child = None

    def cancel(self):
        with self.lock:
            self.closed = True
            child = self.child
        if child is not None:
            self._stop(child)

    def close(self):
        self.cancel()
