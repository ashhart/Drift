"""Raw-row Drift connector for compressed MLA caches, against a stand-in vLLM (no vLLM or GPU needed)."""
from __future__ import annotations
import importlib
import json
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from drift.serving import raw_rows

C4, C128, SWA, IDX = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn", "model.layers.2.self_attn.swa_cache",
                      "model.layers.0.self_attn.indexer.k_cache")


class MLAAttentionSpec:
    def __init__(self, block_size, compress_ratio):
        self.block_size, self.compress_ratio = block_size, compress_ratio


class SlidingWindowMLASpec(MLAAttentionSpec):
    pass


@pytest.fixture
def make(tmp_path, monkeypatch):
    base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorMetadata:
        pass

    class SupportsHMA:
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._meta = None

        def _get_connector_metadata(self):
            return self._meta

    base.KVConnectorMetadata, base.SupportsHMA, base.KVConnectorBase_V1 = KVConnectorMetadata, SupportsHMA, KVConnectorBase_V1
    for name in ("vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
                 "vllm.distributed.kv_transfer.kv_connector.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.delitem(sys.modules, "drift.serving.vllm_raw_rows", raising=False)
    module = importlib.import_module("drift.serving.vllm_raw_rows")
    groups = [NS(layer_names=[C4, IDX], kv_cache_spec=MLAAttentionSpec(256, 4)),         # 64 entries of 584 bytes per block
              NS(layer_names=[C128], kv_cache_spec=MLAAttentionSpec(256, 128)),
              NS(layer_names=[SWA], kv_cache_spec=SlidingWindowMLASpec(64, 1))]
    extra = {"drift_path": str(tmp_path)}
    config = NS(kv_transfer_config=NS(get_from_extra_config=lambda key, default: extra.get(key, default)))

    def new(rank=0):
        monkeypatch.setenv("RANK", str(rank))
        c = module.DriftRawRowConnector(config, "scheduler", NS(num_blocks=6, kv_cache_groups=groups))
        c.register_kv_caches({C4: torch.zeros(6, 64, 584, dtype=torch.uint8), IDX: torch.zeros(6, 64, 132, dtype=torch.uint8),
                              C128: torch.zeros(6, 2, 584, dtype=torch.uint8), SWA: torch.zeros(6, 64, 584, dtype=torch.uint8)})
        return c
    return new, tmp_path


def request(rid, *, inject=None, tap=None, tokens=256, start=256, own=1024, prompt=1100, salt="salt-1"):
    params = {"drift_tokens": tokens, "drift_span_start": start, "drift_own_start": own}
    params.update({k: v for k, v in (("drift_inject", inject), ("drift_tap", tap)) if v})
    return NS(request_id=rid, prompt_token_ids=list(range(prompt)), kv_transfer_params=params, cache_salt=salt, skip_reading_prefix_cache=False)


def step(rid, blocks=((3, 1, 5, 0), (4, 2), ()), computed=0, scheduled=1024):
    return NS(scheduled_new_reqs=[NS(req_id=rid, block_ids=blocks, num_computed_tokens=computed)],
              scheduled_cached_reqs=NS(req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()),
              num_scheduled_tokens={rid: scheduled}, preempted_req_ids=set())


def run(c, rid, **kwargs):
    c._meta = c.build_connector_meta(step(rid, **kwargs))
    c.wait_for_save()
    return c._meta


def marks(root, name):
    return {p.name.removeprefix(f"{name}."): json.loads(p.read_text()) for p in (root / "drift-marks").glob(f"{name}.*")}


def test_only_compressed_mla_caches_are_memory_and_spans_align_to_the_largest_ratio(make):
    new, _ = make
    c = new()
    assert sorted(c._memory) == sorted([C4, IDX, C128]) and c._spans.alignment == 128


def test_a_tapped_span_injects_back_byte_for_byte_and_nothing_else_moves(make):
    new, root = make
    source = new()
    rng = np.random.default_rng(0)
    for name in (C4, IDX, C128, SWA):
        source._kv_caches[name].copy_(torch.from_numpy(rng.integers(0, 256, source._kv_caches[name].shape, dtype=np.uint8)))
    source.on_new_request(request("r0", tap="mem-1"))
    run(source, "r0")
    assert marks(root, "mem-1")["tap.rank0.done"]["tensors"] == 3
    tapped = raw_rows.load(root / "drift-tap" / "mem-1.rank0.npz")
    assert {n: r.shape for n, (_, r) in tapped.items()} == {C4: (64, 584), IDX: (64, 132), C128: (2, 584)}
    np.testing.assert_array_equal(tapped[C4][1], source._kv_caches[C4][1].numpy())             # block 1 is the span's second block
    np.testing.assert_array_equal(tapped[C128][1], source._kv_caches[C128][2].numpy())         # C128: block 2 holds entries 2 and 3
    (root / "drift-inject").mkdir()
    (root / "drift-inject" / "mem-1.npz").write_bytes((root / "drift-tap" / "mem-1.rank0.npz").read_bytes())
    target = new(rank=1)
    before = {name: target._kv_caches[name].clone() for name in (C4, IDX, C128, SWA)}
    target.on_new_request(request("r1", inject="mem-1"))
    meta = run(target, "r1", blocks=((0, 2, 4, 5), (1, 3), ()))
    assert not meta.steps[0].failed and marks(root, "mem-1")["inject.rank1.done"]["tensors"] == 3
    np.testing.assert_array_equal(target._kv_caches[C4][2].numpy(), tapped[C4][1])
    np.testing.assert_array_equal(target._kv_caches[C128][3].numpy(), tapped[C128][1])
    changed = (target._kv_caches[C4] != before[C4]).any(dim=-1)
    assert changed[2].all() and int(changed.sum()) == 64
    assert torch.equal(target._kv_caches[SWA], before[SWA])


def test_one_request_writes_before_it_taps(make):
    new, root = make
    c = new()
    memory = {C4: (4, np.full((64, 584), 7, np.uint8)), IDX: (4, np.full((64, 132), 8, np.uint8)), C128: (128, np.full((2, 584), 9, np.uint8))}
    raw_rows.save(root / "drift-inject" / "mem-2.npz", memory)
    c.on_new_request(request("r2", inject="mem-2", tap="mem-2"))
    meta = run(c, "r2")
    assert [s.kind for s in meta.steps] == ["inject", "tap"]
    readback = raw_rows.load(root / "drift-tap" / "mem-2.rank0.npz")
    assert all(np.array_equal(readback[name][1], memory[name][1]) for name in memory)


@pytest.mark.parametrize("change,reason", [
    ({"start": 64}, "multiple of 128"), ({"salt": None}, "cache_salt"), ({"own": 1100}, "inside the prompt"),
])
def test_invalid_injects_are_refused_on_every_rank(make, change, reason):
    new, root = make
    c = new()
    c.on_new_request(request("r3", inject="mem-3", **change))
    meta = run(c, "r3")
    assert reason in meta.steps[0].failed and reason in marks(root, "mem-3")["inject.rank0.error"]["error"]
    assert not any(cache.any() for cache in c._kv_caches.values())


def test_a_partial_or_misshapen_memory_is_never_written(make):
    new, root = make
    c = new()
    raw_rows.save(root / "drift-inject" / "mem-4.npz", {C4: (4, np.ones((64, 584), np.uint8)), C128: (128, np.ones((2, 584), np.uint8))})
    c.on_new_request(request("r4", inject="mem-4"))
    run(c, "r4")
    assert "partial memory" in marks(root, "mem-4")["inject.rank0.error"]["error"]
    raw_rows.save(root / "drift-inject" / "mem-5.npz", {C4: (4, np.ones((64, 584), np.uint8)), IDX: (4, np.ones((64, 584), np.uint8)),
                                                         C128: (128, np.ones((2, 584), np.uint8))})
    c.on_new_request(request("r5", inject="mem-5", salt="salt-5"))
    run(c, "r5")
    assert "do not fit" in marks(root, "mem-5")["inject.rank0.error"]["error"]
    assert not any(cache.any() for cache in c._kv_caches.values())


def test_a_step_past_own_start_is_refused_but_a_tap_needs_no_boundary(make):
    new, root = make
    c = new()
    raw_rows.save(root / "drift-inject" / "mem-6.npz", {})
    c.on_new_request(request("r6", inject="mem-6", tap="tap-6"))
    meta = run(c, "r6", scheduled=1100)
    kinds = {s.kind: s.failed for s in meta.steps}
    assert "past own start" in kinds["inject"] and kinds["tap"] == ""


def test_a_wrong_page_geometry_fails_closed(make):
    new, root = make
    c = new()
    c._kv_caches[C128] = torch.zeros(6, 4, 584, dtype=torch.uint8)                       # 4 slots where the spec says 2 entries
    c.on_new_request(request("r7", tap="tap-7"))
    run(c, "r7")
    assert "expected 2 entries per block" in marks(root, "tap-7")["tap.rank0.error"]["error"]
