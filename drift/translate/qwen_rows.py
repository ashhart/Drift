"""Qwen3.8-Flash full-attention rows between their flat tap layout and per-layer K/V entries. NumPy only.

A flat row is 1,024 values per full-attention layer, in layer order: that layer's K for its 2 heads of 256, then its V.
Entries are {layer: (K [rows, 2, 256], V [rows, 2, 256])}, the form append_entries and validate_translation take.
"""
from __future__ import annotations
import numpy as np

HEADS, HEAD_DIM = 2, 256
WIDTH = 2 * HEADS * HEAD_DIM


def split_rows(x: np.ndarray, layers) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """[rows, len(layers) x 1024] -> per-layer (K, V) views."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(layers) * WIDTH:
        raise ValueError(f"rows must be [rows, {len(layers) * WIDTH}], got {x.shape}")
    n, half = len(x), HEADS * HEAD_DIM
    return {layer: (x[:, i * WIDTH:i * WIDTH + half].reshape(n, HEADS, HEAD_DIM), x[:, i * WIDTH + half:(i + 1) * WIDTH].reshape(n, HEADS, HEAD_DIM))
            for i, layer in enumerate(layers)}


def join_rows(entries, layers) -> np.ndarray:
    """Per-layer (K, V) -> [rows, len(layers) x 1024], the inverse of split_rows."""
    return np.concatenate([np.concatenate((np.asarray(entries[l][0]).reshape(len(entries[l][0]), -1), np.asarray(entries[l][1]).reshape(len(entries[l][1]), -1)), axis=1)
                           for l in layers], axis=1)


def zero_selector_keys(rows: int, layers, index_dim: int) -> dict[int, np.ndarray]:
    """Selector keys for rows attended densely: with the budget raised past the memory, the selector never picks blocks."""
    return {layer: np.zeros((rows, index_dim), np.float32) for layer in layers}
