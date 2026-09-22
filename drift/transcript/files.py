"""Bounded private reads and immutable, non-overwriting local publications."""
import hashlib
import os
from pathlib import Path
import stat
import tempfile

from drift.serving.worker_activation_files import private_root


def export_parent(value):
    root = Path(value)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError('TRANSCRIPT_EXPORT_PARENT')
    info = root.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError('TRANSCRIPT_EXPORT_PARENT')
    return root


def read(path, maximum):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('TRANSCRIPT_PATH')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('TRANSCRIPT_FILE')
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('TRANSCRIPT_FILE_LIMIT')
    return data


def publish(path, data):
    path = Path(path)
    root = private_root(path.parent)
    fd, name = tempfile.mkstemp(prefix='.transcript-', dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
