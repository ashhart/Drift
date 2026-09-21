"""Read bounded canonical tap files and publish immutable generated-row artifacts."""
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from drift.serving.live_publication import load_publication
from drift.serving.glm_terminal_evidence import terminal_fields
from drift.serving.worker_activation_files import checksum, private_root, snapshot, temporary


def require(condition):
    if not condition: raise ValueError('native outbox contract mismatch')


def read_tap(path, layouts, max_bytes, max_rows, scratch):
    require(path.is_file() and path.resolve() == path and not path.is_symlink())
    size = path.stat().st_size; require(0 < size <= max_bytes)
    target = temporary(scratch)
    try:
        digest = snapshot(path,target,max_bytes)
        with ZipFile(target) as archive:
            require(sum(member.file_size for member in archive.infolist()) <= max_bytes)
        arrays = load_publication(target,layouts,metadata=('start','stop'),max_rows=max_rows)
        require(checksum(path,max_bytes) == digest and target.stat().st_size == size)
        start, stop = arrays.pop('start'), arrays.pop('stop')
        require(stop > start and all(value.shape[0] == stop-start for value in arrays.values()))
        return arrays, {'sha256':digest,'bytes':size,'start':start,'stop':stop}
    finally:
        target.unlink(missing_ok=True)


def write_rows(directory, seq, arrays, maximum):
    require(all(np.isfinite(value).all() and (np.abs(value)<=np.finfo(np.float16).max).all() for value in arrays.values()))
    target = temporary(directory)
    try:
        np.savez(target, **{key:value.astype(np.float16) for key,value in arrays.items()})
        require(target.stat().st_size <= maximum)
        digest = checksum(target,maximum)
        final = directory/f'own-{seq:06d}.npz'
        target.chmod(0o400); os.link(target,final)
        return {'file':final.name,'sha256':digest,'bytes':final.stat().st_size}
    finally:
        target.unlink(missing_ok=True)


def finish_marker(folder):
    private_root(folder)
    require(not any(folder.glob('error.rank*')) and not any(folder.glob('.*.tmp.npz')))
    path = folder/'finished'
    require(path.is_file() and path.resolve() == path and not path.is_symlink() and path.stat().st_size <= 4096)
    result=json.loads(path.read_bytes())
    terminal_fields(result)
    return result


def write_manifest(directory, payload):
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    require(len(raw)<=1048576)
    path=directory/'manifest.json'
    target=temporary(directory)
    try:
        target.write_bytes(raw); target.chmod(0o400); os.link(target,path)
    finally:
        target.unlink(missing_ok=True)
    return {'manifest_path':str(path),'manifest_sha256':hashlib.sha256(raw).hexdigest()}
