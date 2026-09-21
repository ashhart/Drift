"""Bounded pending publications and interruption of controller-owned processes."""
import subprocess
import threading
from collections import deque
from dataclasses import dataclass


class PendingMemoryFull(RuntimeError):
    """A producer exceeded the configured pending-publication capacity."""


@dataclass(frozen=True)
class Delivery:
    path: str
    rows: int


class PendingMemory:
    """Count uploads and unacknowledged appends against the same fixed capacity."""

    def __init__(self, capacity=8):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("pending publication capacity must be a positive integer")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._ready = deque()
        self._pending = {}
        self._peak = 0

    def publish(self, path, rows, upload):
        delivery = Delivery(path, rows)
        with self._lock:
            if len(self._pending) >= self.capacity:
                raise PendingMemoryFull("pending publication capacity exhausted")
            self._pending[id(delivery)] = delivery
            self._peak = max(self._peak, len(self._pending))
        try:
            upload()
        except BaseException:
            self.complete(delivery)
            raise
        with self._lock:
            self._ready.append(delivery)

    def __bool__(self):
        with self._lock:
            return bool(self._ready)

    def take(self):
        with self._lock:
            return self._ready.popleft() if self._ready else None

    def complete(self, delivery):
        with self._lock:
            if self._pending.get(id(delivery)) is not delivery:
                raise ValueError("unknown or already acknowledged publication")
            del self._pending[id(delivery)]

    def report(self):
        with self._lock:
            return {"capacity": self.capacity, "pending": len(self._pending), "peak": self._peak,
                    "pending_rows": sum(item.rows for item in self._pending.values())}


def interrupt_processes(processes, *, graceful=()):
    """Interrupt owned local processes, without claiming remote request cancellation."""
    active = []
    requested = set()
    for proc in processes:
        if proc.poll() is None:
            try:
                if any(proc is target for target in graceful) and getattr(proc, "stdin", None) is not None:
                    try:
                        proc.stdin.write('{"op":"abort"}\n')
                        proc.stdin.flush()
                        proc.stdin.close()
                        requested.add(id(proc))
                    except (OSError, ValueError):
                        proc.terminate()
                else:
                    proc.terminate()
                active.append(proc)
            except OSError:
                pass
    for proc in active:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                if id(proc) in requested:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                        continue
                    except subprocess.TimeoutExpired:
                        pass
                proc.kill()
            except OSError:
                pass
        except OSError:
            pass
