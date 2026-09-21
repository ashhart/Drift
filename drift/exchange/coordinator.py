"""Hold the live exchange sessions for a process and serve one bounded boundary request at a time."""
import json
import os
import select
import socket
import threading
import time

from drift.exchange.session import ExchangeError
from drift.exchange.lifetime import request_scope

FRAME_LIMIT = 4096
MAX_REQUESTS = 1024
PATH_LIMIT = 104                            # the shortest sun_path any supported platform accepts
OPS = {'exchange': {'op', 'route'}, 'status': {'op', 'route'}}


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('EXCHANGE_DUPLICATE')
        value[key] = item
    return value


def _constant(value):
    raise ValueError('EXCHANGE_NUMBER')


def dispatch(sessions, frame, *, check=None):
    """Validate one frame against the op table and run it; every failure is the caller's, never a partial exchange."""
    if type(frame) is not dict:
        raise ValueError('EXCHANGE_FRAME')
    op = frame.get('op')
    if op not in OPS or set(frame) != OPS[op]:
        raise ValueError('EXCHANGE_FRAME')
    route = frame['route']
    if type(route) is not str or route not in sessions:
        raise ValueError('EXCHANGE_ROUTE')
    session = sessions[route]
    if op == 'status':
        return dict(op='status', route=route, mode=session.mode, sequence=session.sequence,
                    foreign_rows=session.foreign_rows, poisoned=session.poisoned)
    with request_scope(check):
        record = session.publish_own()
        applied = session.drain_forward()
    return dict(op='exchanged', route=route, published=_jsonable(record),
                applied=[dict(item) for item in applied],
                foreign_rows=session.foreign_rows)


def _jsonable(record):
    return {key: list(value) if type(value) is tuple else value for key, value in record.items()}


class ExchangeCoordinator:
    """Serialize private connections while retaining route cursors and the lifetime budget."""

    def __init__(self, path, sessions, deadline, max_requests=MAX_REQUESTS):
        self.path, self.sessions, self.deadline = path, dict(sessions), deadline
        self.max_requests, self.requests = max_requests, 0
        self.failed, self.stopped = threading.Event(), threading.Event()
        self.peer = self.thread = None
        if len(os.fsencode(str(path))) >= PATH_LIMIT:
            raise ValueError('EXCHANGE_SOCKET_PATH')
        self.listener = socket.socket(socket.AF_UNIX)
        try:
            self.listener.bind(str(path))
            os.chmod(path, 0o600)
            self.inode = path.stat().st_ino
            self.listener.listen(1)
            self.listener.settimeout(.05)
        except Exception:
            self.listener.close()
            raise

    def start(self):
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _reply(self, frame):
        try:
            return dispatch(self.sessions, frame, check=self._checkpoint)
        except (ExchangeError, ValueError) as error:
            code = error.args[0] if error.args and type(error.args[0]) is str else 'EXCHANGE_FAILED'
            return dict(op='error', code=code)

    def _checkpoint(self):
        if self.stopped.is_set():
            raise ExchangeError('EXCHANGE_CANCELLED')
        if time.monotonic() >= self.deadline:
            raise ExchangeError('EXCHANGE_DEADLINE')
        try:
            readable, _, _ = select.select([self.peer], [], [], 0)
            if readable:
                if not self.peer.recv(1, socket.MSG_PEEK):
                    raise ExchangeError('EXCHANGE_CANCELLED')
                raise ExchangeError('EXCHANGE_FRAME')
        except OSError as error:
            raise ExchangeError('EXCHANGE_CANCELLED') from error

    def _serve(self):
        try:
            while not self.stopped.is_set():
                if time.monotonic() >= self.deadline:
                    raise TimeoutError
                try:
                    self.peer, _ = self.listener.accept()
                except socket.timeout:
                    continue
                try:
                    self._serve_peer()
                finally:
                    self.peer.close()
                    self.peer = None
        except Exception:
            if not self.stopped.is_set():
                self.failed.set()
        finally:
            self.listener.close()
            if self.peer is not None:
                self.peer.close()

    def _serve_peer(self):
        self.peer.settimeout(.05)
        raw = bytearray()
        while not self.stopped.is_set():
            if time.monotonic() >= self.deadline:
                raise TimeoutError
            try:
                block = self.peer.recv(FRAME_LIMIT + 1 - len(raw))
            except socket.timeout:
                continue
            if not block:
                if raw:
                    raise EOFError
                return
            raw.extend(block)
            if len(raw) > FRAME_LIMIT:
                raise ValueError('EXCHANGE_LIMIT')
            if b'\n' not in raw:
                continue
            if raw.count(b'\n') != 1 or not raw.endswith(b'\n') or self.requests >= self.max_requests:
                raise ValueError('EXCHANGE_FRAME')
            frame = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
            raw.clear()
            reply = self._reply(frame)
            self.requests += 1
            self.peer.sendall(json.dumps(reply, allow_nan=False).encode() + b'\n')

    def close(self):
        self.stopped.set()
        self.listener.close()
        peer = self.peer
        if peer is not None:
            try:
                peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=max(0, self.deadline - time.monotonic()))
        if self.path.exists() and self.path.stat().st_ino == self.inode:
            self.path.unlink()
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('EXCHANGE_THREAD_UNCONFIRMED')
