"""The vLLM shim against a fake `vllm` package that mimics the scheduler and worker objects.

This qualifies the shim's LOGIC only (planning, slot mapping, tap, inject, exchange files).
It says nothing about a live server; see docs/SERVING_INTEGRATION.md for that ladder.
"""
from __future__ import annotations
import sys
import types
from types import SimpleNamespace
import pytest
import torch


@pytest.fixture
def connector_module(monkeypatch):
    base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorMetadata:
        pass

    class KVConnectorRole:
        SCHEDULER, WORKER = "scheduler", "worker"

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._role, self._meta = role, None

        def bind_connector_metadata(self, meta):
            self._meta = meta

        def _get_connector_metadata(self):
            return self._meta

    base.KVConnectorMetadata, base.KVConnectorRole, base.KVConnectorBase_V1 = KVConnectorMetadata, KVConnectorRole, KVConnectorBase_V1
    for name in ("vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
                 "vllm.distributed.kv_transfer.kv_connector.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.delitem(sys.modules, "drift.serving.vllm_connector", raising=False)
    import drift.serving.vllm_connector as module
    return module


BLOCK = 4
PAD = 99


def make(module, tmp_path, layout="blocks_first", layers=None, cache_dtype="auto", tp=1):
    extra = {"exchange_root": str(tmp_path), "placeholder_token_id": PAD, "layout": layout,
             "kv_layers": layers or {"L3": 3, "L7": 7}, "rope_theta": 10000.0, "rotary_dim": 4, "tp_rank_override": 0}
    if layout == "mla":
        extra.pop("rope_theta"); extra.pop("rotary_dim")
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype=cache_dtype),
                             parallel_config=SimpleNamespace(tensor_parallel_size=tp),
                             kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda k, d=None: extra.get(k, d)))
    return module.DriftConnector(config, "scheduler"), module.DriftConnector(config, "worker")


def request(rid, prompt, **params):
    return SimpleNamespace(request_id=rid, prompt_token_ids=prompt, kv_transfer_params=params or None)


def run_step(scheduler, worker, req, block_ids, caches, matched):
    scheduler.update_state_after_alloc(req, None, matched)
    meta = scheduler.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[
        SimpleNamespace(req_id=req.request_id, prompt_token_ids=req.prompt_token_ids, block_ids=[block_ids])]))
    worker.bind_connector_metadata(meta)
    ctx = SimpleNamespace(no_compile_layers={name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()},
                          attn_metadata={})
    worker.start_load_kv(ctx)
    for name, cache in caches.items():
        worker.save_kv_layer(name, cache, None)
    worker.wait_for_save()
    return meta


def test_tap_then_inject_through_the_shim_roundtrips(connector_module, tmp_path):
    from drift.core.types import KV
    from drift.serving.exchange import load_entries, save_entries
    scheduler, worker = make(connector_module, tmp_path)
    torch.manual_seed(0)
    caches = {"L3": torch.randn(8, 2, BLOCK, 2, 8), "L7": torch.randn(8, 2, BLOCK, 2, 8)}
    # 1) a stock-looking request that only taps its 6 prompt tokens
    tap_req = request("r1", [5, 6, 7, 8, 9, 10], drift_session="s1", drift_tap="ctx0")
    assert scheduler.get_num_new_matched_tokens(tap_req, 0) == (0, False)
    run_step(scheduler, worker, tap_req, [2, 5], caches, 0)
    entries, positions, manifest = load_entries(tmp_path / "s1" / "tap", "ctx0")
    assert set(entries) == {3, 7} and positions.tolist() == list(range(6)) and manifest["placeholders"] == 0
    # 2) the sidecar would translate; here the same entries (trimmed to a block multiple) go back as foreign memory
    foreign = {k: KV(v.k[:4], v.v[:4]) for k, v in entries.items()}
    save_entries(tmp_path / "s1" / "inject", "mem0", foreign, torch.arange(4), {"epoch": 0})
    inj_req = request("r2", [PAD] * 4 + [11, 12], drift_session="s1", drift_inject="mem0", drift_tap="ctx1")
    matched, _ = scheduler.get_num_new_matched_tokens(inj_req, 0)
    assert matched == 4
    fresh = {name: torch.zeros_like(cache) for name, cache in caches.items()}
    run_step(scheduler, worker, inj_req, [1, 6], fresh, matched)
    # the placeholder slots now hold the foreign entries in NATIVE form (rotated at positions 0..3)
    from drift.serving.core import LayerSpec, RopeSpec, tap_layer
    spec = LayerSpec("L3", 3, "blocks_first", RopeSpec(10000.0, 4))
    got = tap_layer(spec, fresh["L3"], torch.tensor([4, 5, 6, 7]), torch.arange(4))
    torch.testing.assert_close(got.k, foreign[3].k, rtol=1e-5, atol=1e-5)
    assert torch.equal(got.v, foreign[3].v)
    # the tap of the injected request covers only its REAL tokens, at positions after the placeholders
    tapped, tap_positions, tap_manifest = load_entries(tmp_path / "s1" / "tap", "ctx1")
    assert tapped[3].tokens == 2 and tap_manifest["placeholders"] == 4


def test_stock_requests_and_bad_requests(connector_module, tmp_path):
    scheduler, worker = make(connector_module, tmp_path)
    assert scheduler.get_num_new_matched_tokens(request("r", [1, 2, 3]), 0) == (0, False)
    with pytest.raises(ValueError, match="not in the exchange"):
        scheduler.get_num_new_matched_tokens(request("r", [PAD] * 4 + [1], drift_session="s", drift_inject="missing"), 0)
    from drift.serving.exchange import save_entries
    from drift.core.types import KV
    save_entries(tmp_path / "s" / "inject", "odd", {3: KV(torch.randn(3, 2, 8), torch.randn(3, 2, 8)), 7: KV(torch.randn(3, 2, 8), torch.randn(3, 2, 8))},
                 torch.arange(3), {})
    with pytest.raises(ValueError, match="block size"):
        scheduler.get_num_new_matched_tokens(request("r", [PAD] * 3 + [1], drift_session="s", drift_inject="odd"), 0)
    # metadata of a stock step is empty and the worker does nothing
    meta = scheduler.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="x", prompt_token_ids=[1], block_ids=[[0]])]))
    assert meta.requests == []


def test_mla_layout_and_quantized_cache_refusal(connector_module, tmp_path):
    from drift.serving.exchange import load_entries, save_entries
    scheduler, worker = make(connector_module, tmp_path, layout="mla")
    caches = {"L3": torch.zeros(6, BLOCK, 16), "L7": torch.zeros(6, BLOCK, 16)}
    latents = {3: torch.randn(4, 16), 7: torch.randn(4, 16)}
    save_entries(tmp_path / "s" / "inject", "m", latents, torch.arange(4), {})
    req = request("r", [PAD] * 4 + [1, 2], drift_session="s", drift_inject="m", drift_tap="t")
    matched, _ = scheduler.get_num_new_matched_tokens(req, 0)
    run_step(scheduler, worker, req, [3, 0], caches, matched)
    assert torch.equal(caches["L7"].view(24, 16)[torch.tensor([12, 13, 14, 15])], latents[7])
    tapped, _, _ = load_entries(tmp_path / "s" / "tap", "t")
    assert tapped[3].shape == (2, 16)
    with pytest.raises(ValueError, match="not qualified"):
        make(connector_module, tmp_path, layout="mla", cache_dtype="fp8_ds_mla")
