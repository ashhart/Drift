"""Qualify foreign-key rephasing on the deployed oMLX QSA cache. Run with the oMLX runtime; prints one JSON verdict."""
from __future__ import annotations
import json
import sys

import numpy as np

from drift.serving.foreign_positions import ForeignPositionBank
from drift.serving.omlx_cache import Rope

LAYERS, RATIO = (3, 7), 2                     # blocks start at slots 0, 2, 4 and 6, so foreign slots 2 and 6 open blocks
NATIVE = [0, 4, 5]


def load_runtime():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.core as mx
    import mlx_vlm
    from mlx_vlm.models.qwen4_exp import language, qsa_fast
    return mx, language.QSAKVCache, qsa_fast.pool_completed_index_keys, {"mlx": mx.__version__, "mlx_vlm": mlx_vlm.__version__}


def entries(rows):
    key = np.arange(rows * 2 * 8, dtype=np.float32).reshape(rows, 2, 8) / 20
    return {layer: (key + layer, key - layer) for layer in LAYERS}


def check() -> dict:
    mx, QSAKVCache, pool, versions = load_runtime()
    norm = lambda keys: keys
    rotate = lambda keys, positions: keys + positions[:, None, :, None].astype(keys.dtype)
    cache = {}
    for layer in LAYERS:
        current = QSAKVCache()
        current.update_and_fetch(mx.arange(128, dtype=mx.float32).reshape(1, 2, 8, 8), mx.full((1, 2, 8, 8), 3, dtype=mx.float32))
        current.update_indexer(mx.arange(32, dtype=mx.float32).reshape(1, 8, 4) / 7, mx.arange(1000, 1008)[None])
        cache[layer] = current
    stale = {layer: np.array(c.pooled_indexer_keys(RATIO, norm, rotate)) for layer, c in cache.items()}
    before = {layer: [np.array(a) for a in (c.keys, c.values, c.index_keys, c.index_position_ids)] for layer, c in cache.items()}
    bank = ForeignPositionBank(LAYERS, 5, heads=2, head_dim=8)
    bank.remember(entries(3), np.array([100, 100, 101]), 1)
    bank.remember(entries(2), np.array([104, 105]), 6)
    rope, checks = Rope(10000, 8), {}

    def record(name, value):
        checks[name] = checks.get(name, True) and bool(value)

    for query in (200, 201):
        bank.rephase(cache, rope, query)
        positions = np.array([query - 6, query - 6, query - 5, query - 2, query - 1])
        for layer, current in cache.items():
            keys, values, index_keys, index_positions = before[layer]
            expected = rope.apply(mx.array(bank.canonical[layer].transpose(1, 0, 2))[None], positions)
            now = np.array(current.keys)
            record("foreign_keys_rotated_exactly", np.array_equal(now[:, :, bank.slots], np.array(expected)))
            record("native_keys_unchanged", np.array_equal(now[:, :, NATIVE], keys[:, :, NATIVE]))
            record("values_unchanged", np.array_equal(np.array(current.values), values))
            record("selector_keys_unchanged", np.array_equal(np.array(current.index_keys), index_keys))
            record("foreign_selector_positions_updated", np.array_equal(np.array(current.index_position_ids)[:, bank.slots], positions[None]))
            record("native_selector_positions_unchanged", np.array_equal(np.array(current.index_position_ids)[:, NATIVE], index_positions[:, NATIVE]))
            record("offset_unchanged", current.offset == 8)
            pooled = np.array(current.pooled_indexer_keys(RATIO, norm, rotate))
            fresh = np.array(pool(current.index_keys, current.index_position_ids, compress_ratio=RATIO, index_key_norm=norm, apply_index_rope=rotate))
            record("block_bank_matches_a_fresh_pool", np.array_equal(pooled, fresh))
            record("foreign_blocks_rebuilt", not np.array_equal(pooled[:, [1, 3]], stale[layer][:, [1, 3]]))
            record("native_blocks_kept", np.array_equal(pooled[:, [0, 2]], stale[layer][:, [0, 2]]))
    for layer, current in cache.items():
        rephased = np.array(current.index_position_ids)
        current.update_and_fetch(mx.zeros((1, 2, 1, 8), dtype=mx.float32), mx.zeros((1, 2, 1, 8), dtype=mx.float32))
        current.update_indexer(mx.ones((1, 1, 4)), mx.array([[1008]]))
        record("next_token_appends_after_rephase", current.offset == 9 and current.index_keys.shape == (1, 9, 4)
               and np.array_equal(np.array(current.index_position_ids)[:, :8], rephased))
    return {"runtime": versions, "checks": checks, "rephase_calls": bank.rephase_calls, "passed": all(checks.values())}


if __name__ == "__main__":
    verdict = check()
    print(json.dumps(verdict), flush=True)
    sys.exit(0 if verdict["passed"] else 1)
