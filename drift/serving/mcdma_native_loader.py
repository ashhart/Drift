"""Load an operator-pinned MCDMA binding and library without library search."""
import ctypes
import hashlib
import os
from pathlib import Path
import re
import stat
import types


def pinned(spec, maximum):
    if type(spec) is not dict or set(spec) != {'path', 'sha256'}:
        raise ValueError('MCDMA_ARTIFACT_PIN')
    path = Path(spec['path'])
    if not path.is_absolute() or path != path.resolve() or not re.fullmatch('[a-f0-9]{64}', spec['sha256']):
        raise ValueError('MCDMA_ARTIFACT_PIN')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022:
            raise ValueError('MCDMA_ARTIFACT_OWNER')
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('MCDMA_ARTIFACT_PIN')
    return path, raw


def load_opener(*, binding, library):
    path, code = pinned(binding, 1048576)
    native, _ = pinned(library, 64 * 1048576)
    module = types.ModuleType('drift_pinned_mcdma')
    module.__file__ = str(path)
    exec(compile(code, str(path), 'exec'), module.__dict__)
    if not callable(getattr(module, 'open', None)):
        raise ValueError('MCDMA_BINDING_API')
    pinned(library, 64 * 1048576)
    loaded = ctypes.CDLL(str(native))
    module._load = lambda: loaded
    return module.open
