"""Read a completed GLM export locally; raw headers never enter the activation link."""
import json
import mmap
import os
from pathlib import Path
import stat
import time

from drift.serving.glm53_handoff import read_header_bytes, read_latents
from drift.transcript.files import read

MAX_EXPORT = 1024 ** 3


def extract(directory, handoff_id, ids, deadline, *, arena=None):
    directory = Path(directory)
    ready = directory / 'rank0.ready'
    while not ready.exists():
        if (directory / 'error.rank0').exists():
            raise ValueError('TRANSCRIPT_EXPORT_FAILED')
        if time.monotonic() >= deadline:
            raise TimeoutError('TRANSCRIPT_EXPORT_TIMEOUT')
        time.sleep(min(.02, max(0, deadline - time.monotonic())))
    spec = json.loads(read(ready, 65536))
    mode = spec.get('mode')
    path = directory / 'rank0.bin' if mode == 'file' else arena
    if path is None or mode not in ('file', 'arena'):
        raise ValueError('TRANSCRIPT_EXPORT_MODE')
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('TRANSCRIPT_EXPORT_PATH')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        offset = 0 if mode == 'file' else spec.get('offset')
        length = spec.get('bytes') if mode == 'file' else spec.get('length')
        if (not stat.S_ISREG(info.st_mode) or type(offset) is not int or type(length) is not int
                or offset < 0 or not 0 < length <= MAX_EXPORT or offset + length > info.st_size
                or mode == 'arena' and spec.get('arena_inode') != info.st_ino):
            raise ValueError('TRANSCRIPT_EXPORT_BOUNDS')
        with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mapping:
            blob = memoryview(mapping)[offset:offset + length]
            try:
                header, _ = read_header_bytes(blob)
                if (header.get('handoff_id') != handoff_id or header.get('prompt_token_ids') != ids
                        or header.get('n_tokens') != len(ids) or header.get('tp_rank') != 0):
                    raise ValueError('TRANSCRIPT_EXPORT_IDENTITY')
                return read_latents(blob)
            finally:
                blob.release()
    finally:
        os.close(fd)
