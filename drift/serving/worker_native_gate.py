"""Serialize native state changes and GPU completion before publishing receipts."""
import threading
from contextlib import contextmanager
from drift.serving.worker_contract import require


class NativeGate:
    def __init__(self, handle, state, settle, timeout=2):
        self.handle, self.state, self.settle, self.timeout = handle, state, settle, timeout
        self.lock = threading.RLock()
        self.failed = threading.Event()

    def poison(self):
        self.failed.set()
        self.state['poisoned'] = True

    @contextmanager
    def transaction(self):
        acquired = self.lock.acquire(timeout=self.timeout)
        if not acquired:
            self.poison(); require(False, 'LIMIT')
        try:
            require(not self.failed.is_set() and not self.state.get('poisoned'), 'WORKER')
            yield
            require(not self.failed.is_set(), 'WORKER')
        except Exception:
            self.poison()
            raise
        finally:
            self.lock.release()

    def __call__(self, command):
        with self.transaction():
            result = self.handle(command)
            self.settle()
            return result

    def clear(self):
        acquired = self.lock.acquire(timeout=self.timeout)
        if not acquired:
            self.poison(); require(False, 'LIMIT')
        try:
            self.state.clear()
            self.poison()
        finally:
            self.lock.release()
