"""oMLX / mlx-vlm cache shim: tap and inject through the cache, never through attention.

Torch-free on purpose (mlx + numpy) so it runs inside oMLX's bundled runtime unmodified.
Works on the per-layer cache list from `language_model.make_cache()`:
  - KV-bearing layers hold a KVCache-like object with `keys`/`values` [1,H,alloc,D] (keys are
    post-RoPE), `offset`, `update_and_fetch(keys, values)` and, for QSA models,
    `update_indexer(index_keys [1,T,Di], position_ids [1,T])` with `index_keys`.
  - other layers hold recurrent/conv state objects that are left alone.
Connector-mode semantics (docs/SERVING_INTEGRATION.md): foreign entries become a cache PREFIX at
positions 0..N-1; the model's own prompt then runs at positions N.. .
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np


def _mx():
    import mlx.core as mx
    return mx


@dataclass(frozen=True)
class Rope:
    theta: float
    rotary_dim: int

    def _cos_sin(self, positions: np.ndarray):
        mx = _mx()
        inv = 1.0 / (self.theta ** (np.arange(0, self.rotary_dim, 2, dtype=np.float64) / self.rotary_dim))
        phase = positions.astype(np.float64)[:, None] * inv[None, :]
        phase = np.concatenate((phase, phase), axis=-1).astype(np.float32)
        return mx.array(np.cos(phase)), mx.array(np.sin(phase))

    def apply(self, x, positions: np.ndarray, inverse: bool = False):
        """x: mx array [1,H,T,D]; rotates the first `rotary_dim` dims (split-half), float32 math."""
        mx = _mx()
        cos, sin = self._cos_sin(positions)
        if inverse:
            sin = -sin
        cos, sin = cos[None, None], sin[None, None]
        x32 = x.astype(mx.float32)
        rot, rest = x32[..., : self.rotary_dim], x32[..., self.rotary_dim:]
        half = self.rotary_dim // 2
        turned = mx.concatenate((-rot[..., half:], rot[..., :half]), axis=-1)
        return mx.concatenate((rot * cos + turned * sin, rest), axis=-1)


def kv_layer_indices(cache: Sequence[Any]) -> list[int]:
    return [i for i, c in enumerate(cache) if hasattr(c, "keys") and hasattr(c, "update_and_fetch") and hasattr(c, "offset")]


def tap(cache: Sequence[Any], layers: Sequence[int], rope: Rope | None, start: int, stop: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Canonical (de-rotated K, V) as float32 numpy [T,H,D] for positions start..stop-1."""
    mx = _mx()
    out = {}
    positions = np.arange(start, stop)
    for layer in layers:
        c = cache[layer]
        if stop > int(c.offset):
            raise ValueError("tap range exceeds the cache")
        k, v = c.keys[:, :, start:stop], c.values[:, :, start:stop]
        k = rope.apply(k, positions, inverse=True) if rope is not None else k.astype(mx.float32)
        v = v.astype(mx.float32)
        mx.eval(k, v)
        out[layer] = (np.array(k)[0].transpose(1, 0, 2), np.array(v)[0].transpose(1, 0, 2))
    return out


def inject_prefix(cache: Sequence[Any], entries: Mapping[int, tuple[np.ndarray, np.ndarray]], rope: Rope | None,
                  dtype=None, index_keys: Mapping[int, Any] | None = None) -> int:
    """Write canonical entries into FRESH KV caches as positions 0..N-1. Returns N.

    `index_keys` (optional, per layer [1,N,Di]) supplies sparse-selector keys; zeros are used
    otherwise, which is only valid while total context stays within the selector's budget."""
    mx = _mx()
    counts = {k.shape[0] for k, _ in entries.values()}
    if len(counts) != 1:
        raise ValueError("entries must be complete and equal-length across KV layers")
    n = counts.pop()
    positions = np.arange(n)
    for layer, (k, v) in entries.items():
        c = cache[layer]
        if int(c.offset) != 0:
            raise ValueError("prefix injection needs a fresh cache")
        keys = mx.array(np.ascontiguousarray(k.transpose(1, 0, 2)))[None]
        values = mx.array(np.ascontiguousarray(v.transpose(1, 0, 2)))[None]
        keys = rope.apply(keys, positions) if rope is not None else keys
        target = dtype or (c.keys.dtype if getattr(c, "keys", None) is not None else keys.dtype)
        c.update_and_fetch(keys.astype(target), values.astype(target))
        if hasattr(c, "update_indexer"):
            if index_keys is not None and layer in index_keys:
                ik = index_keys[layer]
            else:
                raise ValueError("this cache has a sparse-selector state; pass index_keys (zeros are acceptable within budget)")
            c.update_indexer(ik, mx.array(positions.astype(np.int32))[None])
    return n


def copy_nonkv_state(source: Sequence[Any], target: list, kv_layers: Sequence[int]) -> None:
    """Identity replacement helper: recurrent/conv/n-gram state has no per-token form, so it is copied."""
    for i, c in enumerate(source):
        if i not in kv_layers:
            target[i] = copy.deepcopy(c)


def append_entries(cache: Sequence[Any], entries: Mapping[int, tuple[np.ndarray, np.ndarray]], rope: Rope | None,
                   positions: np.ndarray, index_dim: int | None = None, dtype=None,
                   index_keys: Mapping[int, np.ndarray] | None = None) -> int:
    """Append canonical entries to a RUNNING cache at explicit virtual positions (the spec's foreign-memory append, D16).

    Slot order in the cache is irrelevant to attention; the rotation carries the position. The caller must drive the
    model with explicit `position_ids` from then on, because the cache offset no longer equals the own position.
    Returns the first slot index used."""
    mx = _mx()
    positions = np.asarray(positions)
    counts = {k.shape[0] for k, _ in entries.values()}
    if counts != {len(positions)}:
        raise ValueError("entries must be complete, equal-length and match the positions")
    first = None
    for layer, (k, v) in entries.items():
        c = cache[layer]
        first = int(c.offset) if first is None else first
        if int(c.offset) != first:
            raise ValueError("KV layers disagree about the cache length")
        keys = mx.array(np.ascontiguousarray(k.transpose(1, 0, 2)))[None]
        values = mx.array(np.ascontiguousarray(v.transpose(1, 0, 2)))[None]
        keys = rope.apply(keys, positions) if rope is not None else keys
        target = dtype or (c.keys.dtype if getattr(c, "keys", None) is not None else keys.dtype)
        c.update_and_fetch(keys.astype(target), values.astype(target))
        if hasattr(c, "update_indexer"):
            if index_dim is None:
                raise ValueError("this cache has a sparse-selector state; pass index_dim")
            if index_keys is not None:                             # raw (unrotated) selector keys; beyond the selector budget zeros are never chosen
                raw = np.asarray(index_keys[layer], dtype=np.float32)
                if raw.shape != (len(positions), index_dim):
                    raise ValueError("selector keys must be [rows, index_dim]")
                selector = mx.array(raw)[None].astype(target)
            else:
                selector = mx.zeros((1, len(positions), index_dim), dtype=target)
            c.update_indexer(selector, mx.array(positions.astype(np.int32))[None])
    return int(first)


def tap_slots(cache: Sequence[Any], layers: Sequence[int], rope: Rope | None, slots: np.ndarray, positions: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Canonical entries for arbitrary cache slots whose virtual positions are known (running caches with appends)."""
    mx = _mx()
    index = mx.array(np.asarray(slots, dtype=np.int32))
    out = {}
    for layer in layers:
        c = cache[layer]
        if len(slots) and int(np.max(slots)) >= int(c.offset):
            raise ValueError("tap slots exceed the cache")
        k, v = mx.take(c.keys, index, axis=2), mx.take(c.values, index, axis=2)
        k = rope.apply(k, np.asarray(positions), inverse=True) if rope is not None else k.astype(mx.float32)
        v = v.astype(mx.float32)
        mx.eval(k, v)
        out[layer] = (np.array(k)[0].transpose(1, 0, 2), np.array(v)[0].transpose(1, 0, 2))
    return out
