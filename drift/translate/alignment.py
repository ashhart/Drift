"""Pair two tokenizations of one text by the characters their tokens end on. NumPy only.

Two models split a text differently; where a token of each ends on the same character, the two tokens have read the same
text up to that point, so one can serve as the other's regression target. Offsets are [tokens, 2] (start, end) arrays
in characters of the shared text.
"""
from __future__ import annotations
import numpy as np


def aligned_pairs(source_offsets: np.ndarray, target_offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices (source, target) of tokens ending on the same character, in increasing order."""
    ends = {int(end): i for i, end in enumerate(np.asarray(target_offsets)[:, 1])}
    pairs = [(i, ends[int(end)]) for i, end in enumerate(np.asarray(source_offsets)[:, 1]) if int(end) in ends]
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    source, target = (np.asarray(side, np.int64) for side in zip(*pairs))
    return source, target
