import ast
from types import SimpleNamespace

import numpy as np
import pytest

from drift.serving.foreign_positions import ForeignPositionBank
from drift.serving.omlx_cache import Rope

from test_mcdma_forward import ROOT, drain, publication


def test_actual_drain_anchors_incoming_source_rows_to_next_query(drain):
    observed = []
    drain.env["args"].own_start = 2048
    drain.env["args"].foreign_row_cap = 2048
    drain.env["own_positions"] = list(range(2048, 2317))
    drain.env["fwd_used"] = 301
    drain.env["foreign_bank"] = SimpleNamespace(
        positions=lambda sources, query: query - 1 - (sources[-1] - sources),
        remember=lambda *args: None,
    )
    drain.env["append_entries"] = lambda cache, entries, rope, positions, *args, **kwargs: observed.extend(positions) or 572
    drain.writer.publish(publication())
    drain.call()
    assert observed == [2315, 2316]


def test_actual_step_rephases_all_retained_rows_before_model_call():
    source = ast.parse((ROOT / "scripts/live/studio_mcdma_loop.py").read_text())
    step = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    events = []

    def rephase(cache, rope, start):
        events.append(("rephase", start))

    def model(ids, **kwargs):
        events.append(("model", kwargs["position_ids"].tolist()))
        return SimpleNamespace(logits=np.zeros((1, ids.shape[-1], 3)))

    env = {"np": np, "args": SimpleNamespace(own_start=2048), "position_gap": 0,
           "own_positions": list(range(2048, 2317)), "own_slots": [], "slots_now": lambda: 572,
           "mx": SimpleNamespace(arange=np.arange, array=np.array, int32=np.int32, eval=lambda _: None),
           "lm": model, "cache": object(), "rope": object(),
           "foreign_bank": SimpleNamespace(rephase=rephase)}
    exec(compile(ast.Module(body=[step], type_ignores=[]), "actual_step", "exec"), env)
    env["step"]([1, 2])
    env["step"]([3])
    assert events == [("rephase", 2317), ("model", [[2317, 2318]]),
                      ("rephase", 2319), ("model", [[2319]])]


@pytest.fixture
def numeric_backend(monkeypatch):
    backend = SimpleNamespace(**{name: getattr(np, name) for name in
                                 ("array", "int32", "float32", "all", "isfinite", "cos", "sin", "concatenate")},
                              eval=lambda _: None)
    monkeypatch.setattr("drift.serving.foreign_positions._mx", lambda: backend)
    monkeypatch.setattr("drift.serving.omlx_cache._mx", lambda: backend)
    return backend


class TinyCache:
    def __init__(self, offset=8):
        self.offset = offset
        self.keys = np.arange(1 * 2 * offset * 8, dtype=np.float32).reshape(1, 2, offset, 8)
        self.values = self.keys.copy() + 3
        self.index_keys = np.full((1, offset, 4), 7, np.float32)
        self.index_position_ids = np.arange(1000, 1000 + offset, dtype=np.int32)[None]
        self.index_block_keys = np.ones((1, 1, 2, 4))
        self.index_block_ratio = 4

    def clear_index_blocks(self):
        self.index_block_keys = None
        self.index_block_ratio = None


def entries(rows, layers=(3, 7)):
    key = np.arange(rows * 2 * 8, dtype=np.float32).reshape(rows, 2, 8) / 20
    return {layer: (key + layer, key - layer) for layer in layers}


def test_rephase_preserves_source_ages_duplicates_native_rows_and_values(numeric_backend):
    bank = ForeignPositionBank((3, 7), 5, heads=2, head_dim=8)
    first = entries(3)
    bank.remember(first, np.array([100, 100, 101]), 1)
    bank.remember(entries(2), np.array([104, 105]), 6)
    first[3][0][:] = 999
    cache = {layer: TinyCache() for layer in (3, 7)}
    before = {layer: (c.keys.copy(), c.values.copy(), c.index_keys.copy(), c.index_position_ids.copy()) for layer, c in cache.items()}
    rope = Rope(10000, 8)
    for query in (200, 201):
        bank.rephase(cache, rope, query)
        expected = np.array([query - 6, query - 6, query - 5, query - 2, query - 1])
        for layer, current in cache.items():
            canonical = bank.canonical[layer].transpose(1, 0, 2)[None]
            np.testing.assert_allclose(current.keys[:, :, bank.slots], rope.apply(canonical, expected), rtol=0, atol=0)
            np.testing.assert_array_equal(current.index_position_ids[:, bank.slots], expected[None])
            np.testing.assert_array_equal(current.keys[:, :, [0, 4, 5]], before[layer][0][:, :, [0, 4, 5]])
            np.testing.assert_array_equal(current.values, before[layer][1])
            np.testing.assert_array_equal(current.index_keys, before[layer][2])
            np.testing.assert_array_equal(current.index_position_ids[:, [0, 4, 5]], before[layer][3][:, [0, 4, 5]])
            assert current.index_block_keys is None and current.index_block_ratio is None
            assert current.offset == 8 and not bank.canonical[layer].flags.writeable
    assert bank.rephase_calls == 2 and bank.rephase_seconds >= 0


def test_positions_allow_negative_recency_and_fanout_identity():
    np.testing.assert_array_equal(ForeignPositionBank.positions(np.array([5, 5, 8]), 1), [-3, -3, 0])


def test_duplicate_sources_match_reference_recency_after_fanout():
    import torch
    from drift.core.position import recency_positions
    source, order = np.array([5, 8, 9]), np.array([0, 0, 1, 2, 2, 2])
    expected = recency_positions(torch.tensor(source), 20).numpy()[order]
    np.testing.assert_array_equal(ForeignPositionBank.positions(source[order], 20), expected)


@pytest.mark.parametrize("sources,query", [([2, 1], 5), ([-1, 1], 5), ([1.0, 2.0], 5),
                                           ([1, 2], True), ([1, 2], -1), ([0, 2**40], 1)])
def test_invalid_positions_fail_closed(sources, query):
    with pytest.raises(RuntimeError):
        ForeignPositionBank.positions(np.array(sources), query)


def test_validates_all_layers_before_cache_mutation(numeric_backend):
    bank = ForeignPositionBank((3, 7), 2, heads=2, head_dim=8)
    bank.remember(entries(2), np.array([5, 6]), 1)
    cache = {layer: TinyCache() for layer in (3, 7)}
    before = cache[3].keys.copy()
    cache[7].index_position_ids = np.zeros((1, 7), np.int32)
    with pytest.raises(RuntimeError, match="layout"):
        bank.rephase(cache, Rope(10000, 8), 40)
    np.testing.assert_array_equal(cache[3].keys, before)
    assert cache[3].index_block_keys is not None and bank.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bank.rephase(cache, Rope(10000, 8), 41)


def test_partial_publication_failure_poisons_future_steps(numeric_backend):
    bank = ForeignPositionBank((3, 7), 2, heads=2, head_dim=8)
    bank.remember(entries(2), np.array([5, 6]), 1)
    cache = {layer: TinyCache() for layer in (3, 7)}
    def fail():
        raise RuntimeError("synthetic selector cache failure")
    cache[7].clear_index_blocks = fail
    with pytest.raises(RuntimeError, match="selector cache failure"):
        bank.rephase(cache, Rope(10000, 8), 40)
    assert bank.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bank.rephase(cache, Rope(10000, 8), 41)


@pytest.mark.parametrize("sources,slot", [([6, 7], 3), ([7, 8], 2)])
def test_overlapping_source_or_physical_slots_poison_bank(sources, slot):
    bank = ForeignPositionBank((3, 7), 4, heads=2, head_dim=8)
    bank.remember(entries(2), np.array([5, 6]), 1)
    with pytest.raises(RuntimeError, match="overlap"):
        bank.remember(entries(2), np.array(sources), slot)
    assert bank.poisoned


def test_capacity_remains_bounded():
    bank = ForeignPositionBank((3, 7), 1, heads=2, head_dim=8)
    with pytest.raises(RuntimeError, match="capacity"):
        bank.remember(entries(2), np.array([5, 6]), 1)
    assert bank.poisoned and not len(bank.sources)


def test_nonfinite_rotation_never_mutates_receiver(numeric_backend):
    bank = ForeignPositionBank((3, 7), 2, heads=2, head_dim=8)
    bank.remember(entries(2), np.array([5, 6]), 1)
    cache = {layer: TinyCache() for layer in (3, 7)}
    before = cache[3].keys.copy()
    rope = SimpleNamespace(apply=lambda key, positions: np.full(key.shape, np.nan))
    with pytest.raises(RuntimeError, match="nonfinite"):
        bank.rephase(cache, rope, 40)
    np.testing.assert_array_equal(cache[3].keys, before)
    assert bank.poisoned


def test_native_mlx_qsa_cache_rephasing_preserves_other_state():
    mx = pytest.importorskip("mlx.core")
    language = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
    cache = {}
    for layer in (3, 7):
        current = language.QSAKVCache()
        current.update_and_fetch(mx.arange(128, dtype=mx.float32).reshape(1, 2, 8, 8),
                                 mx.full((1, 2, 8, 8), 3, dtype=mx.float32))
        current.update_indexer(mx.ones((1, 8, 4)), mx.arange(1000, 1008)[None])
        current.index_block_keys, current.index_block_ratio = mx.ones((1, 1, 2, 4)), 4
        assert current.index_position_ids.shape == (1, current.offset)
        assert current.keys.shape[2] > current.offset
        cache[layer] = current
    before = {layer: (np.array(c.keys), np.array(c.values), np.array(c.index_keys), np.array(c.index_position_ids))
              for layer, c in cache.items()}
    bank = ForeignPositionBank((3, 7), 5, heads=2, head_dim=8)
    bank.remember(entries(3), np.array([100, 100, 101]), 1)
    bank.remember(entries(2), np.array([104, 105]), 6)
    rope = Rope(10000, 8)
    for query in (200, 201):
        bank.rephase(cache, rope, query)
        positions = np.array([query - 6, query - 6, query - 5, query - 2, query - 1])
        for layer, current in cache.items():
            expected = rope.apply(mx.array(bank.canonical[layer].transpose(1, 0, 2))[None], positions)
            np.testing.assert_allclose(np.array(current.keys[:, :, mx.array(bank.slots), :]), np.array(expected), rtol=0, atol=0)
            np.testing.assert_array_equal(np.array(current.keys)[:, :, [0, 4, 5]], before[layer][0][:, :, [0, 4, 5]])
            np.testing.assert_array_equal(np.array(current.values), before[layer][1])
            np.testing.assert_array_equal(np.array(current.index_keys), before[layer][2])
            np.testing.assert_array_equal(np.array(current.index_position_ids)[:, bank.slots], positions[None])
            np.testing.assert_array_equal(np.array(current.index_position_ids)[:, [0, 4, 5]], before[layer][3][:, [0, 4, 5]])
            assert current.index_block_keys is None and current.index_block_ratio is None
            assert current.offset == 8


class PooledCache:
    """The oMLX 0.7.0.dev2 layout: no clear_index_blocks, and a pooled block bank that its invalidator drops."""
    def __init__(self, offset=8):
        tiny = TinyCache(offset)
        self.offset, self.keys, self.values = tiny.offset, tiny.keys, tiny.values
        self.index_keys, self.index_position_ids = tiny.index_keys, tiny.index_position_ids
        self._pooled_index_keys, self._pooled_index_offset = np.ones((1, 2, 4)), 2

    def _invalidate_pooled_indexer(self):
        self._pooled_index_keys, self._pooled_index_offset = None, 0


class BareCache(PooledCache):
    _invalidate_pooled_indexer = None


def test_rephase_drops_the_deployed_pooled_block_bank(numeric_backend):
    bank = ForeignPositionBank((3, 7), 2, heads=2, head_dim=8)
    bank.remember(entries(2), np.array([5, 6]), 1)
    cache = {layer: PooledCache() for layer in (3, 7)}
    bank.rephase(cache, Rope(10000, 8), 40)
    for current in cache.values():
        assert current._pooled_index_keys is None and current._pooled_index_offset == 0
        np.testing.assert_array_equal(current.index_position_ids[:, bank.slots], np.array([[38, 39]]))
    assert bank.rephase_calls == 1 and not bank.poisoned


def test_a_receiver_without_a_block_reset_is_refused_before_any_change(numeric_backend):
    bank = ForeignPositionBank((3, 7), 2, heads=2, head_dim=8)
    bank.remember(entries(2), np.array([5, 6]), 1)
    cache = {3: PooledCache(), 7: BareCache()}
    before = {layer: (c.keys.copy(), c.index_position_ids.copy()) for layer, c in cache.items()}
    with pytest.raises(RuntimeError, match="cannot invalidate selector blocks"):
        bank.rephase(cache, Rope(10000, 8), 40)
    assert bank.poisoned and cache[3]._pooled_index_keys is not None
    for layer, current in cache.items():
        np.testing.assert_array_equal(current.keys, before[layer][0])
        np.testing.assert_array_equal(current.index_position_ids, before[layer][1])


def test_native_rephase_on_the_deployed_qsa_cache():
    pytest.importorskip("mlx.core")
    pytest.importorskip("omlx")
    from scripts.live.studio_rephase_check import check
    verdict = check()
    assert verdict["passed"], verdict


def test_a_position_gap_moves_the_step_and_its_rephase_origin_together():
    source = ast.parse((ROOT / "scripts/live/studio_mcdma_loop.py").read_text())
    step = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    events = []

    def model(ids, **kwargs):
        events.append(("model", kwargs["position_ids"].tolist()))
        return SimpleNamespace(logits=np.zeros((1, ids.shape[-1], 3)))

    env = {"np": np, "args": SimpleNamespace(own_start=2048), "position_gap": 300,
           "own_positions": list(range(2048, 2317)), "own_slots": [], "slots_now": lambda: 572,
           "mx": SimpleNamespace(arange=np.arange, array=np.array, int32=np.int32, eval=lambda _: None),
           "lm": model, "cache": object(), "rope": object(),
           "foreign_bank": SimpleNamespace(rephase=lambda cache, rope, start: events.append(("rephase", start)))}
    exec(compile(ast.Module(body=[step], type_ignores=[]), "actual_step", "exec"), env)
    env["step"]([1, 2])
    assert events == [("rephase", 2617), ("model", [[2617, 2618]])]         # own positions resume after GLM's block
    assert env["own_positions"][-2:] == [2617, 2618]
