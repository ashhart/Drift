"""Serialize worker capture with scheduler finalization across processes."""
from contextlib import contextmanager
import fcntl
import os


@contextmanager
def tap_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(folder / '.capture.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)
