"""Read Qwen3.8 handoff export blobs and slice one span's attention rows and compressed selector keys.

Blob: b"Q38HAND1" | u64 header length | header JSON | zero padding to 4096 | tensor bytes, as written by the owner's
Qwen38HandoffConnector. Each tensor entry names its layer, its pages in the request's block order, their shape and
dtype and their offset in the tensor bytes. Page layouts are the ones in qwen38_pages.
"""
from __future__ import annotations
import json
import struct
import numpy as np
try:
    from qwen38_pages import HEAD_DIM, INDEX_DIM, attention_layers, compressed_layers
except ImportError:
    from drift.serving.qwen38_pages import HEAD_DIM, INDEX_DIM, attention_layers, compressed_layers

MAGIC, ALIGN = b"Q38HAND1", 4096


def read_blob(raw) -> tuple[dict, np.ndarray]:
    """Header and tensor bytes of one rank's blob."""
    raw = np.frombuffer(raw, dtype=np.uint8)
    if bytes(raw[:8]) != MAGIC:
        raise ValueError("not a Qwen3.8 handoff blob")
    (length,) = struct.unpack("<Q", bytes(raw[8:16]))
    header = json.loads(bytes(raw[16:16 + length]))
    start = -(-(16 + length) // ALIGN) * ALIGN
    data = raw[start:start + int(header["data_bytes"])]
    if len(data) != int(header["data_bytes"]):
        raise ValueError("blob is shorter than its header says")
    return header, data


def _float(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns to float32."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _pages(header: dict, data: np.ndarray, layer: str) -> np.ndarray:
    entries = [e for e in header["tensors"] if e.get("layer") == layer and "offset" in e]
    if len(entries) != 1 or entries[0]["dtype"] != "bfloat16" or entries[0].get("first_block_index", 0):
        raise ValueError(f"{layer}: expected one whole bf16 tensor entry")
    entry = entries[0]
    chunk = data[entry["offset"]:entry["offset"] + entry["nbytes"]]
    return chunk.view(np.uint16).reshape(entry["shape"])


def span_rows(blobs: list[tuple[dict, np.ndarray]], start: int, count: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Rotated K and V [count, heads, head_dim] per attention layer, heads merged across ranks in rank order."""
    blobs = sorted(blobs, key=lambda blob: blob[0]["tp_rank"])
    if [h["tp_rank"] for h, _ in blobs] != list(range(blobs[0][0]["tp_size"])):
        raise ValueError("need exactly one blob per tensor-parallel rank")
    names = attention_layers(e["layer"] for e in blobs[0][0]["tensors"] if "offset" in e)
    out = {}
    for layer, name in sorted(names.items()):
        keys, values = [], []
        for header, data in blobs:
            pages = _pages(header, data, name)                     # [pages, heads, slots, 2 * head_dim]
            if pages.ndim != 4 or pages.shape[3] != 2 * HEAD_DIM or (start + count) > pages.shape[0] * pages.shape[2]:
                raise ValueError(f"{name}: pages {pages.shape} do not cover positions {start}..{start + count}")
            tokens = pages.transpose(0, 2, 1, 3).reshape(-1, pages.shape[1], 2 * HEAD_DIM)[start:start + count]
            keys.append(_float(tokens[..., :HEAD_DIM]))
            values.append(_float(tokens[..., HEAD_DIM:]))
        out[layer] = (np.concatenate(keys, axis=1), np.concatenate(values, axis=1))
    return out


def span_selector(blob: tuple[dict, np.ndarray], start: int, count: int, ratio: int = 4) -> dict[int, np.ndarray]:
    """Compressed selector keys [count // ratio, index_dim] per layer for the groups inside the span (rank 0 holds them)."""
    header, data = blob
    if start % ratio or count % ratio:
        raise ValueError("the span must start and end on a selector group")
    names = compressed_layers(e["layer"] for e in header["tensors"] if "offset" in e)
    out = {}
    for layer, name in sorted(names.items()):
        pages = _pages(header, data, name)                         # [pages, groups, 1, index_dim]
        if pages.ndim != 4 or pages.shape[2:] != (1, INDEX_DIM):
            raise ValueError(f"{name}: unexpected compressed page shape {pages.shape}")
        groups = pages.reshape(-1, INDEX_DIM)[start // ratio:(start + count) // ratio]
        if len(groups) != count // ratio:
            raise ValueError(f"{name}: pages do not cover the span's groups")
        out[layer] = _float(groups)
    return out
