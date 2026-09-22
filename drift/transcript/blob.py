"""A fixed GLM latent-only frame, with no transcript, token IDs or provenance fields."""
import struct
import numpy as np
from drift.serving.bridge_files import GLM_LAYERS

HEADER = struct.Struct('<8sI')
MAGIC = b'DRIFTS01'
MAX_ROWS = 4096
WIDTH = 512
MAX_BYTES = HEADER.size + MAX_ROWS * len(GLM_LAYERS) * WIDTH * 4


def encode(latents):
    if set(latents) != set(GLM_LAYERS):
        raise ValueError('TRANSCRIPT_LAYERS')
    arrays = [np.asarray(latents[layer], dtype='<f4') for layer in GLM_LAYERS]
    if any(a.ndim != 2 for a in arrays):
        raise ValueError('TRANSCRIPT_LATENTS')
    rows = arrays[0].shape[0]
    if not 0 < rows <= MAX_ROWS or any(a.shape != (rows, WIDTH) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('TRANSCRIPT_LATENTS')
    return HEADER.pack(MAGIC, rows) + b''.join(a.tobytes(order='C') for a in arrays)


def decode(blob, *, max_rows=MAX_ROWS):
    if type(max_rows) is not int or not 0 < max_rows <= MAX_ROWS:
        raise ValueError('TRANSCRIPT_ROW_BUDGET')
    if not HEADER.size < len(blob) <= MAX_BYTES:
        raise ValueError('TRANSCRIPT_BLOB_SIZE')
    magic, rows = HEADER.unpack_from(blob)
    expected = HEADER.size + rows * len(GLM_LAYERS) * WIDTH * 4
    if magic != MAGIC or not 0 < rows <= min(max_rows, MAX_ROWS) or len(blob) != expected:
        raise ValueError('TRANSCRIPT_BLOB_LAYOUT')
    arrays = np.frombuffer(blob, dtype='<f4', offset=HEADER.size).reshape(len(GLM_LAYERS), rows, WIDTH)
    if not np.isfinite(arrays).all():
        raise ValueError('TRANSCRIPT_BLOB_NONFINITE')
    return {layer: arrays[index].copy() for index, layer in enumerate(GLM_LAYERS)}
