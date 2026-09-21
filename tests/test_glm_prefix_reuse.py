"""Opt-in prefix-cache reads for linked GLM sessions: connector guard, preemption refusal and client policy."""
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
from drift.serving.glm_prefix_reuse import linked_extra, reserve_after_system, reusable_tokens, session_salt, warm_request
from drift.serving.glm_restore_prompt import MARKER, MARKER_ID, verify_prefix

LAYERS = [f"language_model.model.layers.{i}.self_attn.attn" for i in (3, 7)]


@pytest.fixture
def connector(tmp_path, monkeypatch):
    base = types.ModuleType("fake_glm53_base_reuse")

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
        def _tp(): return 0, 1
        def _get_connector_metadata(self): return self.meta
        def on_new_request(self, request): pass
        def build_connector_meta(self, scheduler_output): return Glm53HandoffMetadata()
        def request_finished(self, request, block_ids): return False, None
        def request_finished_all_groups(self, request, block_ids): return False, None
        def wait_for_save(self): pass

    base.Glm53HandoffMetadata, base.Glm53HandoffConnector = Glm53HandoffMetadata, Glm53HandoffConnector
    monkeypatch.setitem(sys.modules, "fake_glm53_base_reuse", base)
    monkeypatch.setenv("DRIFT_GLM53_BASE", "fake_glm53_base_reuse")
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)
    yield importlib.import_module("drift.serving.vllm_glm53_inject").DriftGlm53Connector(), tmp_path
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)


def live(rid, *, start, reserve, prompt=400, name="s", no_store=True, **extra):
    params = {"drift_session": name, "drift_reserve": reserve, "drift_reserve_start": start, **extra}
    return NS(request_id=rid, prompt_token_ids=list(range(prompt)), kv_transfer_params=params, skip_reading_prefix_cache=False, skip_writing_prefix_cache=no_store)


def step(new=(), cached=(), scheduled=None, resumed=()):
    return NS(scheduled_new_reqs=[NS(req_id=r, block_ids=b, num_computed_tokens=c) for r, b, c in new],
              scheduled_cached_reqs=NS(req_ids=[r for r, _, _ in cached], new_block_ids=[b for _, b, _ in cached], num_computed_tokens=[c for _, _, c in cached], resumed_req_ids=set(resumed)),
              num_scheduled_tokens=scheduled or {})


def test_default_still_forbids_cache_reads_and_only_the_literal_true_opts_in(connector):
    c, _ = connector
    plain = live("a", start=128, reserve=64); c.on_new_request(plain)
    assert plain.skip_reading_prefix_cache is True
    opted = live("b", start=128, reserve=64, name="t", drift_prefix_reuse=True); c.on_new_request(opted)
    assert opted.skip_reading_prefix_cache is False                       # reads allowed, guarded at first scheduling
    sloppy = live("c", start=128, reserve=64, name="u", drift_prefix_reuse=1); c.on_new_request(sloppy)
    assert sloppy.skip_reading_prefix_cache is True and "literal true" in c._live["c"]["failed"]


def test_a_hit_that_stays_before_the_span_is_accepted_and_memory_is_written(connector):
    c, root = connector
    c.on_new_request(live("r", start=128, reserve=64, drift_prefix_reuse=True))
    (root / "tp-live-in" / "s").mkdir(parents=True)
    np.savez(root / "tp-live-in" / "s" / "000000.npz", **{f"l{i}": np.ones((4, 512), np.float32) for i in (3, 7)})
    meta = c.build_connector_meta(step(new=[("r", ([1, 2, 3, 0],), 128)], scheduled={"r": 272}))   # 128 positions came from the cache: exactly up to the span
    assert meta.live[0].failed == "" and meta.live[0].before == 128 and meta.live[0].apply == (0,)
    c.meta = meta; c.wait_for_save()
    page = c._kv_caches[LAYERS[0]][2].numpy()                              # positions 128..255 live in the request's second block (index 2)
    assert page[:4].any(axis=1).all() and not page[4:].any() and not c._kv_caches[LAYERS[0]][1].any()   # the cached prefix block is untouched


def test_a_hit_reaching_the_span_fails_closed_with_numeric_diagnostics(connector):
    c, root = connector
    c.on_new_request(live("h", start=100, reserve=64, drift_prefix_reuse=True))
    (root / "tp-live-in" / "s").mkdir(parents=True)
    np.savez(root / "tp-live-in" / "s" / "000000.npz", **{f"l{i}": np.ones((4, 512), np.float32) for i in (3, 7)})
    meta = c.build_connector_meta(step(new=[("h", ([1, 2, 3, 0],), 128)], scheduled={"h": 272}))   # a cached block covers positions 100..127 of the span
    assert meta.live[0].failed_code == "PREFIX_HIT" and dict(meta.live[0].failed_details) == {"hit": 128, "start": 100} and meta.live[0].apply == ()
    c.meta = meta
    with pytest.raises(RuntimeError, match="poisoned"):
        c.wait_for_save()
    assert not any(t.any() for t in c._kv_caches.values())
    assert (root / "tp-live-out" / "s" / "error.rank0").read_text().startswith("LiveReceiverError: PREFIX_HIT")


def test_a_preempted_live_request_is_refused_instead_of_continuing_on_placeholders(connector):
    c, root = connector
    c.on_new_request(live("p", start=0, reserve=64))
    assert c.build_connector_meta(step(new=[("p", ([1, 2, 3, 0],), 0)], scheduled={"p": 400})).live[0].failed == ""
    resumed = c.build_connector_meta(step(cached=[("p", ([0, 3, 2, 1],), 0)], scheduled={"p": 400}, resumed={"p"})).live[0]
    assert resumed.failed_code == "PREEMPTED" and dict(resumed.failed_details) == {"computed": 0}
    ordinary = c.build_connector_meta(step(cached=[("p", None, 400)], scheduled={"p": 1})).live[0]
    assert ordinary.failed_code == "PREEMPTED"                              # the refusal latches for the rest of the request


def test_client_policy_places_the_span_after_the_system_text_and_never_stores_linked_turns():
    body = {"model": "m", "messages": [{"role": "system", "content": "You are a careful engineer."}, {"role": "user", "content": "hi"}], "tools": [{"type": "function"}], "max_tokens": 64}
    reserved = reserve_after_system(body, 3)
    assert reserved["messages"][0]["content"] == "You are a careful engineer." + MARKER * 3 and body["messages"][0]["content"].count(MARKER) == 0
    original, with_span = [5, 6, 7, 8, 9, 20, 21], [5, 6, 7, 8, 9, MARKER_ID, MARKER_ID, MARKER_ID, 20, 21]
    assert verify_prefix(original, with_span, 3)["reserve_start"] == 5      # the existing verifier accepts a span that follows own text
    salt = session_salt("0123456789abcdef-session")
    assert salt == session_salt("0123456789abcdef-session") != session_salt("0123456789abcdef-another") and salt.startswith("drift:")
    warm = warm_request(body, salt)
    assert "kv_transfer_params" not in warm and "vllm_xargs" not in warm and warm["max_tokens"] == 1 and warm["messages"][0] == body["messages"][0] and warm["tools"] == body["tools"]
    extra = linked_extra(salt, {"drift_session": "restore-1", "drift_reserve": 3, "drift_reserve_start": 5})
    assert extra["vllm_xargs"] == {"skip_writing_prefix_cache": 1} and extra["kv_transfer_params"]["drift_prefix_reuse"] is True and extra["cache_salt"] == salt
    with pytest.raises(ValueError):
        linked_extra(salt, {"drift_inject": "x"})
    with pytest.raises(ValueError):
        reserve_after_system({"messages": [{"role": "system", "content": "x" + MARKER}]}, 2)
    with pytest.raises(ValueError):
        session_salt("short")


def test_reusable_tokens_is_cut_at_the_span_and_rounded_to_the_engine_unit():
    prefix = list(range(9000))
    assert reusable_tokens(prefix + [1, 2], prefix + [MARKER_ID] * 64 + [3], reserve_start=9000) == 7168   # two 3,584-token units
    assert reusable_tokens(prefix + [1], prefix[:3000] + [99] + prefix[3001:], reserve_start=9000) == 0   # the prompts diverge before one unit
    assert reusable_tokens(prefix, prefix, reserve_start=3583) == 0                                        # a span inside the first unit gains nothing


def test_review_a_resumed_request_arriving_as_new_is_refused(connector):
    c, _ = connector
    c.on_new_request(live("r", start=128, reserve=64, drift_prefix_reuse=True))
    c.build_connector_meta(step(new=[("r", ([1, 2, 3, 0],), 128)], scheduled={"r": 64}))
    resumed = c.build_connector_meta(step(new=[("r", ([0, 3, 2, 1],), 0)], scheduled={"r": 192})).live[0]
    assert resumed.failed_code == "PREEMPTED" and resumed.blocks == ((0, 3, 2, 1),) and resumed.apply == ()


@pytest.mark.parametrize("flag", [False, None, 1])
def test_review_reuse_is_refused_unless_the_engine_resolved_no_store(connector, flag):
    c, _ = connector
    request = live("n", start=128, reserve=64, drift_prefix_reuse=True, no_store=flag)
    c.on_new_request(request)
    refused = c.build_connector_meta(step(new=[("n", ([1, 2, 3, 0],), 0)], scheduled={"n": 400})).live[0]
    assert refused.failed_code == "NO_STORE" and refused.apply == () and request.skip_reading_prefix_cache is True


def test_review_helpers_reject_unscoped_salts_reserved_prompts_and_invalid_spans():
    from drift.serving.glm_prefix_reuse import reusable_tokens
    body, salt = {"model": "m", "messages": [{"role": "system", "content": "Public fixture"}]}, "drift:" + "a" * 32
    span = {"drift_session": "fixture", "drift_reserve": 3, "drift_reserve_start": 128}
    for bad in (None, "", "drift:short", "other:" + "a" * 32):
        with pytest.raises(ValueError):
            warm_request(body, bad)
        with pytest.raises(ValueError):
            linked_extra(bad, span)
    with pytest.raises(ValueError):
        warm_request(reserve_after_system(body, 3), salt)
    for broken in ({**span, "drift_session": None}, {**span, "drift_reserve": -1}, {**span, "drift_reserve_start": -2}, {**span, "drift_reserve": True}, {**span, "drift_session": "a b"}):
        with pytest.raises(ValueError):
            linked_extra(salt, broken)
    for align in (-3584, 0, 3584.0):
        with pytest.raises(ValueError):
            reusable_tokens(list(range(4000)), list(range(4000)), 4000, align=align)
    with pytest.raises(ValueError):
        reusable_tokens([1], [1], -1)
    assert reusable_tokens(list(range(4000)), list(range(4000)), 4000) == 3584 and reusable_tokens(list(range(4000)), list(range(4000)), 3000) == 0
