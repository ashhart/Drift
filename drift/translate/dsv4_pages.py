"""DeepSeek V4 compressed-cache pages: decode raw page bytes into entries and indexer keys, and encode both back.

A compressed-attention page holds one entry per group of tokens (4 or 128 tokens by the layer's compress ratio): first
every entry's 576 data bytes (448 float8_e4m3fn values, then 64 bfloat16 rope values), then every entry's 8 scale bytes
(7 UE8M0 exponents, one per 64 values, and one zero pad byte). An indexer page holds 128 float8_e4m3fn values per entry,
then one float32 scale per entry. Raw rows from the raw-row connector are fixed-size chunks of these pages, so only
whole pages can be decoded. Layout per vLLM's DeepSeek V4 cache_utils (quantize_and_insert_k_cache). NumPy only.
"""
from __future__ import annotations
import numpy as np
from drift.serving.glm53_handoff import E4M3FN

ENTRY_BYTES, DATA_BYTES, SCALE_BYTES = 584, 576, 8
FP8_DIMS, ROPE_DIMS, BLOCK = 448, 64, 64
INDEX_BYTES, INDEX_DIMS = 132, 128
_POSITIVE = E4M3FN[:0x7F]                                            # codes 0x00..0x7E ascending: 0 .. 448


def _pages(rows: np.ndarray, width: int, per_page: int) -> np.ndarray:
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != width:
        raise ValueError(f"rows must be uint8 [n, {width}]")
    if per_page < 1 or len(rows) % per_page:
        raise ValueError("rows must cover whole pages")
    return rows.reshape(-1, per_page * width)


def entries(rows: np.ndarray, per_page: int) -> np.ndarray:
    """uint8 [n, 584] raw rows of whole pages -> float32 [n, 512] entries: 448 dequantized values, then 64 rope values."""
    pages = _pages(rows, ENTRY_BYTES, per_page)
    data = pages[:, :per_page * DATA_BYTES].reshape(-1, DATA_BYTES)
    scales = pages[:, per_page * DATA_BYTES:].reshape(-1, SCALE_BYTES)
    if scales[:, FP8_DIMS // BLOCK:].any():
        raise ValueError("scale pad bytes are not zero: the rows are not whole, aligned pages")
    factors = np.exp2(scales[:, :FP8_DIMS // BLOCK].astype(np.float32) - 127.0)
    values = E4M3FN[data[:, :FP8_DIMS]].reshape(-1, FP8_DIMS // BLOCK, BLOCK) * factors[:, :, None]
    rope = (np.ascontiguousarray(data[:, FP8_DIMS:]).view("<u2").astype(np.uint32) << 16).view(np.float32)
    out = np.concatenate((values.reshape(-1, FP8_DIMS), rope), axis=1)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite entry after decoding")
    return out


def encode_entries(values: np.ndarray, per_page: int) -> np.ndarray:
    """float [n, 512] entries of whole pages -> uint8 [n, 584] raw rows; the inverse of `entries` up to rounding."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FP8_DIMS + ROPE_DIMS or not np.isfinite(x).all():
        raise ValueError("entries must be finite [n, 512]")
    if per_page < 1 or len(x) % per_page:
        raise ValueError("entries must fill whole pages")
    tiles = x[:, :FP8_DIMS].reshape(len(x), -1, BLOCK)
    peak = np.maximum(np.abs(tiles).max(-1) / 448.0, np.float32(2.0 ** -127))
    exponent = np.clip(np.ceil(np.log2(peak)), -127, 127)             # a power-of-two scale at or above the block's peak / 448
    scaled = np.clip(np.abs(tiles) / np.exp2(exponent)[:, :, None], 0.0, 448.0)
    upper = np.clip(np.searchsorted(_POSITIVE, scaled), 1, len(_POSITIVE) - 1)
    lower = upper - 1
    codes = np.where(scaled - _POSITIVE[lower] <= _POSITIVE[upper] - scaled, lower, upper).astype(np.uint8)
    codes |= np.signbit(tiles).astype(np.uint8) << 7
    bits = np.ascontiguousarray(x[:, FP8_DIMS:]).view(np.uint32)
    rope = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")      # round to nearest even bfloat16
    data = np.concatenate((codes.reshape(len(x), -1), rope.view(np.uint8).reshape(len(x), -1)), axis=1)
    scales = np.zeros((len(x), SCALE_BYTES), np.uint8)
    scales[:, :FP8_DIMS // BLOCK] = (exponent + 127).astype(np.uint8)
    pages = np.concatenate((data.reshape(-1, per_page * DATA_BYTES), scales.reshape(-1, per_page * SCALE_BYTES)), axis=1)
    return pages.reshape(-1, ENTRY_BYTES)


def index_keys(rows: np.ndarray, per_page: int) -> np.ndarray:
    """uint8 [n, 132] raw indexer rows of whole pages -> float32 [n, 128] keys."""
    pages = _pages(rows, INDEX_BYTES, per_page)
    data = pages[:, :per_page * INDEX_DIMS].reshape(-1, INDEX_DIMS)
    scales = np.ascontiguousarray(pages[:, per_page * INDEX_DIMS:]).view("<f4").reshape(-1)
    if not (np.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError("indexer scales must be positive and finite: the rows are not whole, aligned pages")
    out = E4M3FN[data] * scales[:, None]
    if not np.isfinite(out).all():
        raise ValueError("nonfinite indexer key after decoding")
    return out


def encode_index_keys(keys: np.ndarray, per_page: int) -> np.ndarray:
    """float [n, 128] indexer keys of whole pages -> uint8 [n, 132] raw rows; the inverse of `index_keys` up to rounding."""
    x = np.asarray(keys, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != INDEX_DIMS or not np.isfinite(x).all():
        raise ValueError("keys must be finite [n, 128]")
    if per_page < 1 or len(x) % per_page:
        raise ValueError("keys must fill whole pages")
    scales = np.maximum(np.abs(x).max(1) / 448.0, np.float32(2.0 ** -126)).astype("<f4")
    scaled = np.clip(np.abs(x) / scales[:, None], 0.0, 448.0)
    upper = np.clip(np.searchsorted(_POSITIVE, scaled), 1, len(_POSITIVE) - 1)
    lower = upper - 1
    codes = np.where(scaled - _POSITIVE[lower] <= _POSITIVE[upper] - scaled, lower, upper).astype(np.uint8)
    codes |= np.signbit(x).astype(np.uint8) << 7
    pages = np.concatenate((codes.reshape(-1, per_page * INDEX_DIMS), scales.view(np.uint8).reshape(-1, per_page * 4)), axis=1)
    return pages.reshape(-1, INDEX_BYTES)

