"""Rephase retained foreign keys by source age without changing native rows."""
import copy
import time

import numpy as np

from drift.serving.omlx_cache import _mx


def require(value, message):
    if not value:
        raise RuntimeError(message)


class ForeignPositionBank:
    def __init__(self, layers, max_rows, *, heads=2, head_dim=256):
        self.layers = tuple(layers)
        require(self.layers and len(set(self.layers)) == len(self.layers), "invalid foreign layer set")
        require(type(max_rows) is int and max_rows > 0, "invalid foreign row bound")
        self.max_rows, self.heads, self.head_dim = max_rows, heads, head_dim
        self.sources, self.slots = np.empty(0, np.int64), np.empty(0, np.int64)
        self.canonical = {}
        self.poisoned, self.rephase_calls, self.rephase_seconds = False, 0, 0.0

    @staticmethod
    def positions(sources, first_query):
        source = np.asarray(sources)
        require(source.ndim == 1 and source.size > 0 and source.dtype.kind in "iu", "invalid foreign source positions")
        require((source >= 0).all() and (source <= np.iinfo(np.int64).max).all(), "foreign source positions out of range")
        source = source.astype(np.int64)
        require((source[1:] >= source[:-1]).all(), "foreign source positions must preserve fan-out order")
        require(type(first_query) is int and 0 <= first_query <= np.iinfo(np.int32).max, "invalid receiver query position")
        result = first_query - 1 - (source[-1] - source)
        require((result >= np.iinfo(np.int32).min).all(), "foreign positions exceed receiver integer range")
        return result

    def remember(self, entries, sources, first_slot):
        require(not self.poisoned, "foreign position bank poisoned; start a new session")
        try:
            self.positions(sources, 0)
            source = np.array(sources, dtype=np.int64, copy=True)
            rows = len(source)
            require(len(self.sources) + rows <= self.max_rows, "foreign position bank capacity exhausted")
            require(type(first_slot) is int and 0 <= first_slot <= np.iinfo(np.int32).max - rows, "invalid foreign physical slot")
            require(not len(self.slots) or first_slot > self.slots[-1], "foreign physical slots overlap")
            require(not len(self.sources) or source[0] > self.sources[-1], "foreign source intervals overlap")
            require(set(entries) == set(self.layers), "foreign canonical layer set mismatch")
            staged = {}
            for layer in self.layers:
                pair = entries[layer]
                require(isinstance(pair, (tuple, list)) and len(pair) == 2, "invalid foreign canonical KV pair")
                key = np.array(pair[0], copy=True)
                require(key.shape == (rows, self.heads, self.head_dim) and key.dtype.kind == "f", "invalid foreign canonical key shape")
                require(np.isfinite(key).all(), "nonfinite foreign canonical keys")
                if layer in self.canonical:
                    key = np.concatenate((self.canonical[layer], key))
                key.setflags(write=False)
                staged[layer] = key
            slots = np.concatenate((self.slots, np.arange(first_slot, first_slot + rows, dtype=np.int64)))
            source = np.concatenate((self.sources, source))
            slots.setflags(write=False)
            source.setflags(write=False)
            self.canonical, self.sources, self.slots = staged, source, slots
        except Exception:
            self.poisoned = True
            raise

    def rephase(self, cache, rope, first_query):
        require(not self.poisoned, "foreign position bank poisoned; start a new session")
        if not len(self.sources):
            return
        started = time.perf_counter()
        try:
            positions = self.positions(self.sources, first_query)
            mx = _mx()
            slots = mx.array(self.slots, dtype=mx.int32)
            staged, offsets = [], set()
            for layer in self.layers:
                current = cache[layer]
                offset = int(current.offset)
                offsets.add(offset)
                require(offset > int(self.slots[-1]), "foreign slots exceed receiver cache")
                shape = current.keys.shape
                require(len(shape) == 4 and shape[:2] == (1, self.heads) and shape[2] >= offset
                        and shape[3] == self.head_dim, "receiver key cache layout changed")
                require(current.index_position_ids.shape == (1, offset), "receiver selector position layout changed")
                require(callable(current.clear_index_blocks), "receiver cannot invalidate selector blocks")
                canonical = mx.array(np.ascontiguousarray(self.canonical[layer].transpose(1, 0, 2)))[None]
                rotated = rope.apply(canonical, positions).astype(current.keys.dtype)
                finite = mx.all(mx.isfinite(rotated))
                keys = copy.copy(current.keys)
                keys[:, :, slots, :] = rotated
                selector_positions = copy.copy(current.index_position_ids)
                selector_positions[:, slots] = mx.array(positions, dtype=selector_positions.dtype)[None]
                staged.append((current, keys, selector_positions, finite))
            require(len(offsets) == 1, "receiver cache layers disagree on length")
            mx.eval([value for _, keys, positions, finite in staged for value in (keys, positions, finite)])
            require(all(bool(finite.item()) for _, _, _, finite in staged), "nonfinite rephased foreign keys")
            for current, keys, positions, _ in staged:
                current.keys, current.index_position_ids = keys, positions
                current.clear_index_blocks()
            self.rephase_calls += 1
            self.rephase_seconds += time.perf_counter() - started
        except Exception:
            self.poisoned = True
            raise
