"""Validate live file publications before translation or peer delivery.

This checks structure and finite values, not authentication, semantic validity or
epoch causality. Limits are engineering bounds, not calibrated research policy.
"""
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np


MAX_BYTES = 128 * 1024 * 1024


def count(value, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("publication count must be an integer")
    value = int(value)
    if value < int(positive):
        raise ValueError("publication count outside allowed range")
    return value


def load_publication(path, layouts, *, metadata=(), max_rows=4096):
    """Exact layer set, bounded NPY headers, matching row counts, numeric arrays only."""
    expected = set(layouts) | set(metadata)
    if Path(path).stat().st_size > MAX_BYTES:
        raise ValueError("publication archive too large")
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != len(expected) or {m.filename for m in members} != {k + ".npy" for k in expected}:
            raise ValueError("publication fields do not match the declared layout")
        if sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError("publication expands beyond the byte limit")
        arrays = {}
        rows = None
        for member in members:
            key = member.filename[:-4]
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError("unsupported publication array format")
                if key in metadata:
                    if shape != () or dtype.kind not in "iu":
                        raise ValueError("publication metadata must be scalar integers")
                else:
                    if (len(shape) != len(layouts[key]) + 1 or shape[1:] != tuple(layouts[key])
                            or not 0 < shape[0] <= max_rows or dtype.kind != "f"):
                        raise ValueError("publication array shape or dtype mismatch")
                    if rows is not None and rows != shape[0]:
                        raise ValueError("publication layers disagree on row count")
                    rows = shape[0]
                if math.prod(shape) * dtype.itemsize != member.file_size - source.tell():
                    raise ValueError("publication array byte count mismatch")
            with archive.open(member) as source:
                array = np.lib.format.read_array(source, allow_pickle=False)
            if key in metadata:
                arrays[key] = count(array.item())
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    array = array.astype(np.float32)
                if not np.isfinite(array).all():
                    raise ValueError("nonfinite publication entries")
                arrays[key] = array
    return arrays


def wire_arrays(arrays, layouts, rows):
    """Validate every translated layer before converting any output to float16."""
    if set(arrays) != set(layouts):
        raise ValueError("translated layer set mismatch")
    result = {}
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.shape != (rows, *layouts[key]) or value.dtype.kind != "f":
            raise ValueError("translated array shape or dtype mismatch")
        if not np.isfinite(value).all() or (np.abs(value) > np.finfo(np.float16).max).any():
            raise ValueError("translated entries cannot be represented as finite float16")
        result[key] = value.astype(np.float16)
    return result
