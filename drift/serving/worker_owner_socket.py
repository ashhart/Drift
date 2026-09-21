"""Accept one owner-only typed activation connection outside the worker's own stdio."""
import json
import os
import socket
import threading
import time


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value: raise ValueError('OWNER_DUPLICATE')
        value[key] = item
    return value


def _constant(value):
    raise ValueError('OWNER_NUMBER')


class OwnerSocket:
    def __init__(self, path, deadline):
        self.path, self.deadline = path, deadline
        self.failed = threading.Event(); self.stopped = threading.Event()
        self.listener = socket.socket(socket.AF_UNIX)
        self.peer = self.thread = None
        self.requests = 0
        self.busy = threading.Event()
        try:
            self.listener.bind(str(path)); os.chmod(path, 0o600)
            self.inode = path.stat().st_ino
            self.listener.listen(1); self.listener.settimeout(.05)
        except Exception:
            self.listener.close()
            raise

    def start(self, client):
        self.client = client
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _dispatch(self, frame):
        if type(frame) is not dict: raise ValueError('OWNER_FRAME')
        if frame == {'op': 'abort'}: raise ValueError('OWNER_ABORT')
        op = frame.get('op')
        fields = {'op', 'path', 'sha256', 'rows'} if op == 'append' else {'op', 'path', 'after'} if op == 'snapshot_last_own' else {'op', 'path', 'first', 'max_rows'}
        if op not in ('append', 'tap', 'snapshot_last_own') or set(frame) != fields: raise ValueError('OWNER_FRAME')
        if op == 'append': return self.client.append(frame['path'], frame['sha256'], frame['rows'])
        if op == 'snapshot_last_own': return self.client.snapshot_last_own(frame['path'], frame['after'])
        return self.client.tap(frame['path'], frame['first'], frame['max_rows'])

    def _serve(self):
        try:
            while not self.stopped.is_set():
                if time.monotonic() >= self.deadline: raise TimeoutError
                try: self.peer, _ = self.listener.accept(); break
                except socket.timeout: pass
            if self.peer is None: return
            self.listener.close(); self.peer.settimeout(.05)
            raw = bytearray()
            while not self.stopped.is_set():
                if time.monotonic() >= self.deadline: raise TimeoutError
                try: block = self.peer.recv(4097 - len(raw))
                except socket.timeout: continue
                if not block: raise EOFError
                raw.extend(block)
                if len(raw) > 4096: raise ValueError('OWNER_LIMIT')
                if b'\n' not in raw: continue
                if raw.count(b'\n') != 1 or not raw.endswith(b'\n') or self.requests >= 1024:
                    raise ValueError('OWNER_FRAME')
                frame = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant); raw.clear()
                self.busy.set()
                try: result = self._dispatch(frame)
                finally: self.busy.clear()
                if time.monotonic() >= self.deadline: raise TimeoutError
                self.requests += 1
                self.peer.sendall(json.dumps(result, allow_nan=False).encode() + b'\n')
        except Exception:
            if not self.stopped.is_set(): self.failed.set()
        finally:
            if self.peer is not None: self.peer.close()

    def close(self):
        self.stopped.set(); self.listener.close()
        connections = [self.peer]
        if self.busy.is_set(): connections.append(getattr(self, 'client', None))
        for connection in connections:
            connection = getattr(connection, 'socket', connection)
            if connection is not None:
                try: connection.shutdown(socket.SHUT_RDWR)
                except OSError: pass
        if self.thread is not None:
            self.thread.join(timeout=max(0, self.deadline - time.monotonic()))
        if self.path.exists() and self.path.stat().st_ino == self.inode: self.path.unlink()
        if self.thread is not None and self.thread.is_alive(): raise RuntimeError('OWNER_THREAD_UNCONFIRMED')
