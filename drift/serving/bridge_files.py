"""Snapshot controller-private sources and publish bounded immutable bridge files."""
from contextlib import contextmanager
import os
import math
from pathlib import Path
import re
from zipfile import ZipFile
import numpy as np
from drift.serving.worker_activation_files import bounded_path, checksum, private_root, snapshot, temporary

GLM_LAYERS = tuple(3+4*i for i in range(11))
QWEN_LAYERS = tuple(3+4*i for i in range(12))
GLM_LAYOUT = {f'l{layer}': (512,) for layer in GLM_LAYERS}
QWEN_LAYOUT = {f'{kind}{layer}': (2,256) for layer in QWEN_LAYERS for kind in ('k','v')}


def require(value):
    if not value: raise ValueError('BRIDGE_CONTRACT_FAILED')


def integer(value, maximum, minimum=1):
    require(type(value) is int and minimum <= value <= maximum)
    return value


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value))
    return value


def canonical(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file())
    return path


@contextmanager
def frozen(path, expected, scratch, maximum):
    path=canonical(path);pin(expected);integer(maximum,1024*1048576)
    target=temporary(private_root(scratch))
    try:
        require(snapshot(path,target,maximum)==expected)
        target.chmod(0o400)
        yield target
        require(checksum(path,maximum)==expected and checksum(target,maximum)==expected)
    finally:
        target.unlink(missing_ok=True)


def archive_bound(path, maximum):
    require(path.stat().st_size <= maximum)
    with ZipFile(path) as archive: require(sum(item.file_size for item in archive.infolist())<=maximum)


def publish(output, arrays, maximum):
    output=Path(output);root=private_root(output.parent);bounded_path(root,output.name)
    require(not output.exists());integer(maximum,128*1048576)
    target=temporary(root)
    try:
        np.savez(target,**arrays);archive_bound(target,maximum)
        digest=checksum(target,maximum);target.chmod(0o400);os.link(target,output)
        return {'path':str(output),'sha256':digest,'bytes':output.stat().st_size}
    finally:
        target.unlink(missing_ok=True)


def output_bound(layouts, rows, maximum):
    require(4096+sum(rows*math.prod(shape)*2+1024 for shape in layouts.values())<=maximum)
