"""Qwen3.8 Drift injection connector, against a stand-in for the owner's base connector (no vLLM needed)."""
from __future__ import annotations
import importlib
import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from drift.serving.qwen38_pages import rotate

ATTN = {layer: f"language_model.model.layers.{layer}.self_attn.attn" for layer in (3, 7)}
SELECT = {layer: f"language_model.model.layers.{layer}.self_attn.indexer.compressed_key_cache" for layer in (3, 7)}
MTP = "mtp.layers.48.self_attn.attn"


@pytest.fixture
def make(tmp_path, monkeypatch):
    base = types.ModuleType("fake_qwen38_base")

    @dataclass
    class Qwen38HandoffMetadata:
        requests: list = field(default_factory=list)

    class Qwen38HandoffConnector:
        def __init__(self, rank=0):
            self._output_path, self._num_blocks, self.rank = tmp_path, 6, rank
            self._kv_caches = {name: torch.zeros(6, 1, 8, 512, dtype=torch.bfloat16) for name in (*ATTN.values(), MTP)}
            self._kv_caches.update({name: torch.zeros(6, 2, 1, 128, dtype=torch.bfloat16) for name in SELECT.values()})
            self._layer_to_group = {**{name: 0 for name in (*ATTN.values(), MTP)}, **{name: 1 for name in SELECT.values()}}
            self._kv_transfer_config = NS(get_from_extra_config=lambda key, default: default)
            text = NS(head_dim=256, rope_theta=10_000_000.0, partial_rotary_factor=0.25)
            self._vllm_config = NS(model_config=NS(hf_config=NS(text_config=text)))
            self.meta, self.saved, self.seen = None, 0, []
        def _page_rows(self, tensor, blocks):
            return [int(b) for b in blocks], 1
        def _tp(self): return self.rank, 2
        def _get_connector_metadata(self): return self.meta
        def on_new_request(self, request): self.seen.append(request.request_id)
        def build_connector_meta(self, scheduler_output): return Qwen38HandoffMetadata(requests=["owner-request"])
        def request_finished(self, request, block_ids): return False, None
        def request_finished_all_groups(self, request, block_ids): return False, None
        def wait_for_save(self): self.saved += 1

    base.Qwen38HandoffMetadata, base.Qwen38HandoffConnector = Qwen38HandoffMetadata, Qwen38HandoffConnector
    monkeypatch.setitem(sys.modules, "fake_qwen38_base", base)
    monkeypatch.setenv("DRIFT_QWEN38_BASE", "fake_qwen38_base")
    sys.modules.pop("drift.serving.vllm_qwen38_drift", None)
    module = importlib.import_module("drift.serving.vllm_qwen38_drift")
    yield lambda rank=0: module.DriftQwen38Connector(rank), tmp_path
    sys.modules.pop("drift.serving.vllm_qwen38_drift", None)


def request(rid, *, tokens=8, start=4, own=12, prompt=20, name="mem-1", salt="salt-1", **extra):
    params = {"drift_inject": name, "drift_tokens": tokens, "drift_span_start": start, "drift_own_start": own, **extra}
    return NS(request_id=rid, prompt_token_ids=list(range(prompt)), kv_transfer_params=params, cache_salt=salt, skip_reading_prefix_cache=False)


def step(new=(), cached=(), scheduled=None, preempted=()):
    return NS(scheduled_new_reqs=[NS(req_id=r, block_ids=b, num_computed_tokens=c) for r, b, c in new],
              scheduled_cached_reqs=NS(req_ids=[r for r, _, _ in cached], new_block_ids=[b for _, b, _ in cached],
                                       num_computed_tokens=[c for _, _, c in cached], resumed_req_ids=set()),
              num_scheduled_tokens=scheduled or {}, preempted_req_ids=set(preempted))


def memory(root, name="mem-1", rows=8, selector=True, layers=(3, 7), rotated=False, seed=0):
    rng = np.random.default_rng(seed)
    entries = {}
    for layer in layers:
        entries[f"{'kr' if rotated else 'k'}{layer}"] = rng.standard_normal((rows, 2, 256)).astype(np.float32)
        entries[f"v{layer}"] = rng.standard_normal((rows, 2, 256)).astype(np.float32)
        if selector:
            entries[f"c{layer}"] = rng.standard_normal((rows // 4, 128)).astype(np.float32)
    (root / "drift-inject").mkdir(exist_ok=True)
    np.savez(root / "drift-inject" / f"{name}.npz", **entries)
    return entries


def bf16(values):
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16)


def ready(c, rid="r1", blocks=((2, 5), (4, 1)), computed=0, scheduled=12):
    meta = c.build_connector_meta(step(new=[(rid, blocks, computed)], scheduled={rid: scheduled}))
    c.meta = meta
    return meta


@pytest.mark.parametrize("rank", [0, 1])
def test_memory_lands_in_this_rank_head_after_the_placeholder_step(make, rank):
    new, root = make
    c = new(rank)
    entries = memory(root)
    r = request("r1")
    c.on_new_request(r)
    assert r.skip_reading_prefix_cache is True and c.seen == ["r1"]
    meta = ready(c)
    assert meta.requests == ["owner-request"] and len(meta.inject) == 1 and not meta.inject[0].failed
    c.wait_for_save()
    report = json.loads((root / "drift-inject" / f"mem-1.rank{rank}.done").read_text())
    assert report["layers"] == 2 and report["selector"] is True and c.saved == 1
    positions = np.arange(4, 12)
    pages, slots = np.array([2, 2, 2, 2, 5, 5, 5, 5]), np.array([4, 5, 6, 7, 0, 1, 2, 3])
    for layer, name in ATTN.items():
        cache = c._kv_caches[name]
        keys = rotate(entries[f"k{layer}"][:, rank:rank + 1], positions, 1e7, 64)[:, 0]
        assert torch.equal(cache[pages, 0, slots, :256], bf16(keys))
        assert torch.equal(cache[pages, 0, slots, 256:], bf16(entries[f"v{layer}"][:, rank]))
        written = torch.zeros(6, 8, dtype=torch.bool); written[pages, slots] = True
        assert not cache[:, 0][~written].any()
        selector = c._kv_caches[SELECT[layer]]
        assert torch.equal(selector[[4, 1], [1, 0], 0], bf16(entries[f"c{layer}"]))
        assert int((selector != 0).any(dim=-1).sum()) == 2
    assert not c._kv_caches[MTP].any()


def test_pre_rotated_keys_are_written_as_given(make):
    new, root = make
    c = new()
    entries = memory(root, rotated=True, selector=False)
    c.on_new_request(request("r1"))
    ready(c)
    c.wait_for_save()
    assert json.loads((root / "drift-inject" / "mem-1.rank0.done").read_text())["selector"] is False
    assert torch.equal(c._kv_caches[ATTN[3]][[2, 5], 0, [4, 0], :256], bf16(entries["kr3"][[0, 4], 0]))
    assert not any(c._kv_caches[name].any() for name in SELECT.values())


@pytest.mark.parametrize("change,reason", [
    ({"salt": None}, "cache_salt"),
    ({"start": 2}, "multiple of 4"),
    ({"tokens": 6}, "multiple of 4"),
    ({"own": 20}, "own start"),
    ({"name": "../escape"}, "must match"),
    ({"drift_tokens": "8"}, "integer"),
])
def test_invalid_requests_write_an_error_and_touch_nothing(make, change, reason):
    new, root = make
    c = new()
    memory(root)
    args = {k: v for k, v in change.items() if not k.startswith("drift_")}
    r = request("r1", **args)
    r.kv_transfer_params.update({k: v for k, v in change.items() if k.startswith("drift_")})
    c.on_new_request(r)
    meta = ready(c)
    assert reason in meta.inject[0].failed
    c.wait_for_save()
    assert not (root / "drift-inject" / "mem-1.rank0.done").exists() or change.get("name")
    errors = list((root / "drift-inject").glob("*.rank0.error"))
    assert len(errors) == 1 and reason in json.loads(errors[0].read_text())["error"]
    assert not any(cache.any() for cache in c._kv_caches.values())


def test_a_reused_salt_is_refused(make):
    new, root = make
    c = new()
    memory(root)
    c.on_new_request(request("r0", salt="shared"))
    c.on_new_request(request("r1", salt="shared", name="mem-2"))
    meta = c.build_connector_meta(step(new=[("r0", ((2, 5), (4, 1)), 0), ("r1", ((0, 3), (2, 3)), 0)], scheduled={"r0": 12, "r1": 12}))
    failed = {s.request_id: s.failed for s in meta.inject}
    assert failed["r0"] == "" and "cache_salt" in failed["r1"]


def test_a_step_past_own_start_is_refused(make):
    new, root = make
    c = new()
    memory(root)
    c.on_new_request(request("r1"))
    meta = ready(c, scheduled=16)
    assert "past own start" in meta.inject[0].failed
    c.wait_for_save()
    assert (root / "drift-inject" / "mem-1.rank0.error").exists() and not any(cache.any() for cache in c._kv_caches.values())


def test_placeholders_split_over_steps_wait_for_the_last_one(make):
    new, root = make
    c = new()
    memory(root)
    c.on_new_request(request("r1"))
    first = c.build_connector_meta(step(new=[("r1", ((2,), (4,)), 0)], scheduled={"r1": 8}))
    assert first.inject == []
    second = c.build_connector_meta(step(cached=[("r1", ((5,), (1,)), 8)], scheduled={"r1": 4}))
    assert len(second.inject) == 1 and not second.inject[0].failed and second.inject[0].block_ids == ((2, 5), (4, 1))


def test_a_partial_memory_is_never_written(make):
    new, root = make
    c = new()
    memory(root, layers=(3,))
    c.on_new_request(request("r1"))
    ready(c)
    c.wait_for_save()
    assert "missing layers [7]" in json.loads((root / "drift-inject" / "mem-1.rank0.error").read_text())["error"]
    assert not any(cache.any() for cache in c._kv_caches.values())


def test_preemption_after_the_write_is_reported(make):
    new, root = make
    c = new()
    memory(root)
    c.on_new_request(request("r1"))
    ready(c)
    c.build_connector_meta(step(preempted=["r1"]))
    assert json.loads((root / "drift-inject" / "mem-1.preempted").read_text()) == {"request_id": "r1"}


def test_stock_requests_pass_through(make):
    new, root = make
    c = new()
    stock = NS(request_id="s1", prompt_token_ids=list(range(20)), kv_transfer_params=None, cache_salt=None, skip_reading_prefix_cache=False)
    c.on_new_request(stock)
    meta = c.build_connector_meta(step(new=[("s1", ((2,), (4,)), 0)], scheduled={"s1": 20}))
    c.meta = meta
    c.wait_for_save()
    assert stock.skip_reading_prefix_cache is False and meta.inject == [] and c.saved == 1
    assert not (root / "drift-inject").exists()


def test_rotation_matches_complex_multiplication():
    rng = np.random.default_rng(3)
    keys, positions = rng.standard_normal((5, 2, 16)).astype(np.float32), np.array([0, 1, 7, 100, 4097])
    out = rotate(keys, positions, 10_000.0, 8)
    inv = 1.0 / (10_000.0 ** (np.arange(0, 8, 2) / 8))
    turn = np.exp(1j * positions[:, None] * inv[None, :])[:, None, :]
    expected = (keys[..., :4] + 1j * keys[..., 4:8]) * turn
    np.testing.assert_allclose(out[..., :4], expected.real, atol=2e-5)
    np.testing.assert_allclose(out[..., 4:8], expected.imag, atol=2e-5)
    np.testing.assert_array_equal(out[..., 8:], keys[..., 8:])


def test_a_prefix_hit_over_the_span_is_refused(make):
    new, root = make
    c = new()
    memory(root)
    c.on_new_request(request("r1"))
    meta = ready(c, computed=8, scheduled=4)
    assert "prefix-cache hit of 8" in meta.inject[0].failed
