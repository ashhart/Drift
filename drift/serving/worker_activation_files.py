"""Constrain activation files to a private root and verify immutable local snapshots."""
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
from zipfile import ZipFile
from drift.serving.worker_contract import require
from drift.serving.live_publication import load_publication


def private_root(value):
    root = Path(value)
    require(root.is_dir() and root.absolute() == root.resolve(), 'CAPABILITY')
    info = root.stat()
    require(info.st_uid == os.getuid() and not info.st_mode & 0o077, 'CAPABILITY')
    return root


def bounded_path(root, name):
    require(type(name) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.npz', name) is not None)
    path = root / name
    require(not path.is_symlink() and path.resolve().parent == root, 'CAPABILITY')
    return path


def temporary(root):
    fd, name = tempfile.mkstemp(prefix='private-', suffix='.npz', dir=root)
    os.close(fd)
    return Path(name)


def snapshot(source, target, maximum):
    checksum, total = hashlib.sha256(), 0
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as incoming, target.open('wb') as outgoing:
        require(stat.S_ISREG(os.fstat(incoming.fileno()).st_mode), 'CAPABILITY')
        while chunk := incoming.read(min(65536, maximum - total + 1)):
            total += len(chunk)
            require(total <= maximum, 'LIMIT')
            checksum.update(chunk); outgoing.write(chunk)
        outgoing.flush(); os.fsync(outgoing.fileno())
    return checksum.hexdigest()


def publication(path, layouts, maximum, max_rows):
    require(path.stat().st_size <= maximum, 'LIMIT')
    with ZipFile(path) as archive:
        require(sum(member.file_size for member in archive.infolist()) <= maximum, 'LIMIT')
    arrays = load_publication(path, layouts, max_rows=max_rows)
    return next(iter(arrays.values())).shape[0]


def checksum(path, maximum):
    result, total = hashlib.sha256(), 0
    with path.open('rb') as handle:
        while chunk := handle.read(min(65536, maximum - total + 1)):
            total += len(chunk); require(total <= maximum, 'LIMIT'); result.update(chunk)
    return result.hexdigest()
