"""Qwen -> GLM injection connector, against a stand-in for the owner's base connector (no vLLM needed)."""
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
from drift.serving.glm53_handoff import dequantize_fp8_ds_mla

LAYERS = [f"language_model.model.layers.{i}.self_attn.attn" for i in (3, 7)]


@pytest.fixture
def connector(tmp_path, monkeypatch):
    base = types.ModuleType("fake_glm53_base")

    @dataclass
    class Glm53HandoffMetadata:
        requests: list = field(default_factory=list)

    class Glm53HandoffConnector:
        def __init__(self):
            self._output_path, self._mla_group, self._num_blocks = tmp_path, 1, 6
            self._kv_caches = {name: torch.zeros(12, 4, 656, dtype=torch.uint8) for name in LAYERS}      # 2 kernel pages per block
            self._kv_caches["model.layers.45.self_attn.attn"] = torch.zeros(12, 4, 656, dtype=torch.uint8)
            self._layer_to_group = {name: 1 for name in self._kv_caches}
            self.meta, self.saved, self.seen = None, 0, []
            self._kv_transfer_config = NS(get_from_extra_config=lambda key, default: default)
        def _page_rows(self, tensor, blocks):
            ratio = tensor.shape[0] // self._num_blocks
            return [b * ratio + p for b in blocks for p in range(ratio)], ratio
        @staticmethod
        def _tp(): return 0, 2
        def _get_connector_metadata(self): return self.meta
        def on_new_request(self, request): self.seen.append(request.request_id)
        def build_connector_meta(self, scheduler_output): return Glm53HandoffMetadata(requests=["owner-request"])
        def request_finished(self, request, block_ids): return False, None
        def request_finished_all_groups(self, request, block_ids): return False, None
        def wait_for_save(self): self.saved += 1

    base.Glm53HandoffMetadata, base.Glm53HandoffConnector = Glm53HandoffMetadata, Glm53HandoffConnector
    monkeypatch.setitem(sys.modules, "fake_glm53_base", base)
    monkeypatch.setenv("DRIFT_GLM53_BASE", "fake_glm53_base")
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)
    module = importlib.import_module("drift.serving.vllm_glm53_inject")
    owner = module.DriftGlm53Connector()
    owner._drift_rank_failed = bool
    yield owner, tmp_path
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)


def request(rid, tokens, prompt=12, name="mem-1", own_start=None):
    params = {"drift_inject": name, "drift_tokens": tokens, **({"drift_own_start": own_start} if own_start else {})}
    return NS(request_id=rid, prompt_token_ids=list(range(prompt)), kv_transfer_params=params, skip_reading_prefix_cache=False)


def step(new=(), cached=(), scheduled=None):
    return NS(scheduled_new_reqs=[NS(req_id=r, block_ids=b, num_computed_tokens=c) for r, b, c in new],
              scheduled_cached_reqs=NS(req_ids=[r for r, _, _ in cached], new_block_ids=[b for _, b, _ in cached], num_computed_tokens=[c for _, _, c in cached], resumed_req_ids=set()),
              num_scheduled_tokens=scheduled or {})


def test_latents_are_overwritten_after_the_placeholder_step_and_read_back(connector):
    c, root = connector
    rng = np.random.default_rng(0)
    entries = {f"l{i}": rng.standard_normal((6, 512)).astype(np.float32) for i in (3, 7)}
    (root / "tp-inject").mkdir(); np.savez(root / "tp-inject" / "mem-1.npz", **entries)
    r = request("r1", tokens=6)
    c.on_new_request(r)
    assert r.skip_reading_prefix_cache is True and c.seen == ["r1"]
    meta = c.build_connector_meta(step(new=[("r1", ([0], [4, 2]), 0)], scheduled={"r1": 6}))          # step ends exactly at the boundary
    assert meta.requests == ["owner-request"] and len(meta.inject) == 1 and not meta.inject[0].failed and meta.inject[0].block_ids[1] == (4, 2)
    before = c._kv_caches["model.layers.45.self_attn.attn"].clone()
    c.meta = meta; c.wait_for_save()
    assert c.saved == 1 and json.loads((root / "tp-inject" / "mem-1.rank0.done").read_text())["layers"] == 2
    page_rows = [8, 9, 4, 5]                                                                             # block 4 -> rows 8,9; block 2 -> rows 4,5
    for i, name in zip((3, 7), LAYERS):
        flat = c._kv_caches[name][page_rows].reshape(-1, 656).numpy()
        back = dequantize_fp8_ds_mla(flat[:6])
        assert np.sqrt(((back - entries[f"l{i}"]) ** 2).mean()) / np.sqrt((entries[f"l{i}"] ** 2).mean()) < 0.03
        assert not flat[6:].any() and not c._kv_caches[name][[0, 1, 2, 3, 6, 7, 10, 11]].any()            # nothing outside the 6 positions
    assert torch.equal(c._kv_caches["model.layers.45.self_attn.attn"], before)                           # drafter cache untouched
    assert c.build_connector_meta(step()).inject == []                                                    # emitted once


def test_a_step_that_overshoots_the_boundary_fails_closed(connector):
    c, root = connector
    (root / "tp-inject").mkdir(); np.savez(root / "tp-inject" / "mem-1.npz", l3=np.zeros((6, 512), np.float32), l7=np.zeros((6, 512), np.float32))
    c.on_new_request(request("r2", tokens=6))
    meta = c.build_connector_meta(step(new=[("r2", ([0], [1, 2]), 0)], scheduled={"r2": 12}))           # whole prompt in one step
    assert "past the placeholder boundary" in meta.inject[0].failed
    c.meta = meta
    with pytest.raises(RuntimeError, match="poisoned"):
        c.wait_for_save()
    assert "placeholder boundary" in json.loads((root / "tp-inject" / "mem-1.rank0.error").read_text())["error"]
    assert not any(t.any() for t in c._kv_caches.values())


def test_chunked_placeholders_wait_for_the_boundary_and_partial_memories_are_refused(connector):
    c, root = connector
    (root / "tp-inject").mkdir(); np.savez(root / "tp-inject" / "mem-1.npz", l3=np.ones((6, 512), np.float32))   # layer 7 missing
    c.on_new_request(request("r3", tokens=6))
    assert c.build_connector_meta(step(new=[("r3", ([0], [1]), 0)], scheduled={"r3": 4})).inject == []
    meta = c.build_connector_meta(step(cached=[("r3", ([], [3]), 4)], scheduled={"r3": 2}))
    assert len(meta.inject) == 1 and not meta.inject[0].failed and meta.inject[0].block_ids[1] == (1, 3)
    c.meta = meta
    with pytest.raises(RuntimeError, match="poisoned"):
        c.wait_for_save()
    assert "missing layers [7]" in json.loads((root / "tp-inject" / "mem-1.rank0.error").read_text())["error"]
    assert not any(t.any() for t in c._kv_caches.values())
    bad = request("r4", tokens=12)
    c.on_new_request(bad)
    assert "leave at least one own token" in c._tp_pending["r4"].failed
    c.request_finished(bad, [])
    assert "r4" not in c._tp_pending


def test_a_step_ending_inside_the_sacrificial_span_writes_only_the_memory_positions(connector):
    c, root = connector
    (root / "tp-inject").mkdir(); np.savez(root / "tp-inject" / "mem-1.npz", l3=np.ones((4, 512), np.float32), l7=np.ones((4, 512), np.float32))
    c.on_new_request(request("r5", tokens=4, own_start=8))                                              # positions 4..7 keep their own latents
    meta = c.build_connector_meta(step(new=[("r5", ([0], [5, 1]), 0)], scheduled={"r5": 8}))            # the step ends at 8 == own_start
    assert len(meta.inject) == 1 and not meta.inject[0].failed
    c.meta = meta; c.wait_for_save()
    flat = c._kv_caches[LAYERS[0]][[10, 11]].reshape(-1, 656)                                            # block 5 -> rows 10, 11
    assert flat[:4].any(dim=1).all() and not flat[4:].any()
    c.on_new_request(request("r6", tokens=4, own_start=8))
    assert "past the placeholder boundary" in c.build_connector_meta(step(new=[("r6", ([0], [2, 3]), 0)], scheduled={"r6": 9})).inject[0].failed
    assert "drift_own_start" in (lambda r: (c.on_new_request(r), c._tp_pending["r7"].failed)[1])(request("r7", tokens=8, own_start=4))


def test_short_memories_are_tiled_across_the_span(connector):
    c, root = connector
    rng = np.random.default_rng(5)
    rows = {f"l{i}": rng.standard_normal((3, 512)).astype(np.float32) for i in (3, 7)}
    (root / "tp-inject").mkdir(); np.savez(root / "tp-inject" / "mem-1.npz", **rows)
    c.on_new_request(request("r8", tokens=8, own_start=8))
    c.meta = c.build_connector_meta(step(new=[("r8", ([0], [0, 3]), 0)], scheduled={"r8": 8})); c.wait_for_save()
    pages = c._kv_caches[LAYERS[1]][[0, 1, 6, 7]].reshape(-1, 656).numpy()
    assert not pages[8:].any()
    np.testing.assert_allclose(dequantize_fp8_ds_mla(pages[:8]), dequantize_fp8_ds_mla(np.concatenate((__import__("drift.serving.glm53_handoff", fromlist=["x"]).pack_fp8_ds_mla(rows["l7"][np.arange(8) % 3]), np.zeros((8, 128), np.uint8)), 1)))


def live_request(rid, reserve, prompt=200, name="sess-1"):
    return NS(request_id=rid, prompt_token_ids=list(range(prompt)), kv_transfer_params={"drift_session": name, "drift_reserve": reserve}, skip_reading_prefix_cache=False,
              num_in_flight_tokens=0, num_stale_output_tokens=0)


def test_live_session_writes_progressively_and_taps_only_verified_own_positions(tmp_path, monkeypatch):
    import importlib, sys, types
    from dataclasses import dataclass, field
    base = types.ModuleType("fake_glm53_base_live")

    @dataclass
    class Glm53HandoffMetadata:
        requests: list = field(default_factory=list)

    class Glm53HandoffConnector:
        def __init__(self):
            self._output_path, self._mla_group, self._num_blocks = tmp_path, 0, 4
            self._kv_caches = {name: torch.zeros(4, 128, 656, dtype=torch.uint8) for name in LAYERS}
            self._layer_to_group = {name: 0 for name in LAYERS}
            self._kv_transfer_config = NS(get_from_extra_config=lambda key, default: default)
            self.meta = None
        def _page_rows(self, tensor, blocks): return list(blocks), 1
        @staticmethod
        def _tp(): return 0, 2
        def _get_connector_metadata(self): return self.meta
        def on_new_request(self, request): pass
        def build_connector_meta(self, scheduler_output): return Glm53HandoffMetadata()
        def request_finished(self, request, block_ids): return False, None
        def request_finished_all_groups(self, request, block_ids): return False, None
        def wait_for_save(self): pass

    base.Glm53HandoffMetadata, base.Glm53HandoffConnector = Glm53HandoffMetadata, Glm53HandoffConnector
    monkeypatch.setitem(sys.modules, "fake_glm53_base_live", base)
    monkeypatch.setenv("DRIFT_GLM53_BASE", "fake_glm53_base_live")
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)
    c = importlib.import_module("drift.serving.vllm_glm53_inject").DriftGlm53Connector()
    c._drift_rank_failed = bool
    rng = np.random.default_rng(7)
    # the model's own computed latents for positions 0..255 (2 blocks of 128): pack known values so taps can be checked
    own = {i: rng.standard_normal((256, 512)).astype(np.float32) for i in (3, 7)}
    from drift.serving.glm53_handoff import pack_fp8_ds_mla
    for i, name in zip((3, 7), LAYERS):
        c._kv_caches[name][[2, 0]] = torch.from_numpy(np.concatenate((pack_fp8_ds_mla(own[i]), np.zeros((256, 128), np.uint8)), 1).reshape(2, 128, 656))
    request = live_request("L1", reserve=64)
    c.on_new_request(request)
    assert request.skip_reading_prefix_cache is True
    # step 1: prefill of 136 positions; one memory file is already waiting
    memory = {f"l{i}": rng.standard_normal((5, 512)).astype(np.float32) for i in (3, 7)}
    (tmp_path / "tp-live-in" / "sess-1").mkdir(parents=True); np.savez(tmp_path / "tp-live-in" / "sess-1" / "000000.npz", **memory)
    meta = c.build_connector_meta(step(new=[("L1", ([2, 0],), 0)], scheduled={"L1": 136}))
    assert meta.live[0].apply == (0,) and meta.live[0].blocks == ((2, 0),) and (meta.live[0].before, meta.live[0].after) == (0, 136)
    c.meta = meta; c.wait_for_save()
    got = dequantize_fp8_ds_mla(c._kv_caches[LAYERS[1]][2].numpy()[:5])
    assert np.sqrt(((got - memory["l7"]) ** 2).mean()) / np.sqrt((memory["l7"] ** 2).mean()) < 0.03            # reserved slots 0..4 now hold the memory
    assert not list((tmp_path / "tp-live-out").glob("sess-1/*.npz"))                                           # nothing verified beyond the reserve yet
    # step 2: decode; positions 0..135 are verified -> own positions 64..135 are tapped, the reserved span is not
    c.meta = c.build_connector_meta(step(cached=[("L1", None, 136)], scheduled={"L1": 8})); c.wait_for_save()
    tap = np.load(tmp_path / "tp-live-out" / "sess-1" / "000000.npz")
    assert (int(tap["start"]), int(tap["stop"])) == (64, 136) and tap["l3"].shape == (72, 512)
    expected = dequantize_fp8_ds_mla(np.concatenate((pack_fp8_ds_mla(own[3][64:136]), np.zeros((72, 128), np.uint8)), 1))
    np.testing.assert_allclose(tap["l3"].astype(np.float32), expected.astype(np.float16).astype(np.float32), rtol=1e-3, atol=1e-3)
    # step 3: fewer than drift_tap_every new verified positions -> no tap; a second memory lands behind the first
    np.savez(tmp_path / "tp-live-in" / "sess-1" / "000001.npz", **{k: v[:3] for k, v in memory.items()})
    c.meta = c.build_connector_meta(step(cached=[("L1", None, 140)], scheduled={"L1": 8})); c.wait_for_save()
    assert c._live_worker["sess-1"] == {"filled": 8, "tapped": 136, "taps": 1, "next_sequence": 2}
    assert np.allclose(dequantize_fp8_ds_mla(c._kv_caches[LAYERS[0]][2].numpy()[5:8]), dequantize_fp8_ds_mla(np.concatenate((pack_fp8_ds_mla(memory["l3"][:3]), np.zeros((3, 128), np.uint8)), 1)))
    # overflow of the reserved span fails closed with an error file; finishing writes the marker
    np.savez(tmp_path / "tp-live-in" / "sess-1" / "000002.npz", **{k: np.zeros((60, 512), np.float32) for k in memory})
    c.meta = c.build_connector_meta(step(cached=[("L1", None, 150)], scheduled={"L1": 8}))
    with pytest.raises(RuntimeError, match="poisoned"):
        c.wait_for_save()
    assert "reserved span exhausted" in (tmp_path / "tp-live-out" / "sess-1" / "error.rank0").read_text()
    c.request_finished(request, [])
    assert json.loads((tmp_path / "tp-live-out" / "sess-1" / "finished").read_text())["writes_scheduled"] == 3
    bad = live_request("L2", reserve=500)
    c.on_new_request(bad)
    assert "must lie inside the prompt" in c._live["L2"]["failed"]
    # a span that starts inside the prompt: writes land at start+filled, taps begin after the span
    framed = live_request('L3', reserve=40, name='sess-3')
    framed.kv_transfer_params['drift_reserve_start'] = 10
    c.on_new_request(framed)
    (tmp_path / "tp-live-in" / "sess-3").mkdir(parents=True); np.savez(tmp_path / "tp-live-in" / "sess-3" / "000000.npz", **{k: v[:2] for k, v in memory.items()})
    c.meta = c.build_connector_meta(step(new=[("L3", ([1],), 0)], scheduled={"L3": 100})); c.wait_for_save()
    rows = c._kv_caches[LAYERS[0]][1].numpy()
    assert rows[10:12].any(axis=1).all() and not rows[:10].any() and not rows[12:50].any()
    c.meta = c.build_connector_meta(step(cached=[("L3", None, 100)], scheduled={"L3": 4})); c.wait_for_save()
    assert int(np.load(tmp_path / "tp-live-out" / "sess-3" / "000000.npz")["start"]) == 50
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)
