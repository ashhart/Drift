"""GLM query capture: flagged prompt rows become latent queries q W_UK on every rank, and anything ambiguous is refused."""
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
from drift.serving import glm_query_capture as qc

LAYERS = [f"language_model.model.layers.{i}.self_attn.attn" for i in (3, 7)]


class FakeMLA:
    def __init__(self, name, heads=2, nope=4, latent=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.layer_name, self.scale, self.qk_nope_head_dim = name, 0.0625, nope
        self.W_UK_T = torch.randn(heads, nope, latent, generator=g)
        self.calls = 0

    def forward_impl(self, q, *args, **kwargs):
        self.calls += 1
        return q.sum()


@pytest.fixture
def connector(tmp_path, monkeypatch):
    base = types.ModuleType("fake_glm53_base_capture")

    @dataclass
    class Glm53HandoffMetadata:
        requests: list = field(default_factory=list)

    class Glm53HandoffConnector:
        def __init__(self, capture=True, rank=0):
            self._output_path, self._mla_group, self._num_blocks, self.rank = tmp_path, 1, 6, rank
            self._kv_caches = {name: torch.zeros(12, 4, 656, dtype=torch.uint8) for name in LAYERS}
            self._layer_to_group = {name: 1 for name in LAYERS}
            self.meta, self.loaded = None, 0
            extra = {"drift_query_capture": capture}
            self._kv_transfer_config = NS(get_from_extra_config=lambda key, default: extra.get(key, default))
        def _tp(self): return self.rank, 2
        def _get_connector_metadata(self): return self.meta
        def on_new_request(self, request): pass
        def build_connector_meta(self, scheduler_output): return Glm53HandoffMetadata(requests=[])
        def register_kv_caches(self, kv_caches): pass
        def start_load_kv(self, forward_context, **kwargs): self.loaded += 1
        def request_finished(self, request, block_ids): return False, None
        def request_finished_all_groups(self, request, block_ids): return False, None
        def wait_for_save(self): pass

    base.Glm53HandoffMetadata, base.Glm53HandoffConnector = Glm53HandoffMetadata, Glm53HandoffConnector
    monkeypatch.setitem(sys.modules, "fake_glm53_base_capture", base)
    monkeypatch.setenv("DRIFT_GLM53_BASE", "fake_glm53_base_capture")
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)
    module = importlib.import_module("drift.serving.vllm_glm53_inject")
    qc.CAPTURE.clear()
    yield module, tmp_path
    qc.CAPTURE.clear()
    sys.modules.pop("drift.serving.vllm_glm53_inject", None)


def request(rid, rows=(5, 6, 8), prompt=12, name="cap-1"):
    params = {"drift_capture": name, "drift_capture_rows": list(rows)}
    return NS(request_id=rid, prompt_token_ids=list(range(prompt)), kv_transfer_params=params, skip_reading_prefix_cache=False)


def step(rid, computed=0, scheduled=12, total=None):
    return NS(scheduled_new_reqs=[NS(req_id=rid, block_ids=([0], [1, 2]), num_computed_tokens=computed)],
              scheduled_cached_reqs=NS(req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()),
              num_scheduled_tokens={rid: scheduled}, total_num_scheduled_tokens=scheduled if total is None else total)


def test_latent_queries_are_q_times_w_uk_per_head():
    layer = FakeMLA("x", heads=3, nope=4, latent=5)
    q = torch.randn(7, 3, 6)                                          # a rope part beyond the NoPE dims is ignored
    got, scale = qc.latent_queries((q, layer.W_UK_T, 0.5, 4), [2, 3, 6])
    expected = np.stack([q[[2, 3, 6], h, :4].numpy() @ layer.W_UK_T[h].numpy() for h in range(3)], axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert scale == 0.5 and got.shape == (3, 3, 5)
    with pytest.raises(ValueError):
        qc.latent_queries((q, layer.W_UK_T, 0.5, 4), [5, 7])


@pytest.mark.parametrize("rank", [0, 1])
def test_a_flagged_step_writes_each_rank_s_latent_queries(connector, rank):
    module, root = connector
    c = module.DriftGlm53Connector(rank=rank)
    c._drift_rank_failed = bool
    layers = {name: FakeMLA(name, seed=i) for i, name in enumerate(LAYERS)}
    assert qc.install(FakeMLA)
    r = request("r1")
    c.on_new_request(r)
    assert r.skip_reading_prefix_cache is True
    c.meta = c.build_connector_meta(step("r1"))
    assert [(s.name, s.positions, s.before, s.failed) for s in c.meta.capture] == [("cap-1", (5, 6, 8), 0, "")]
    c.start_load_kv(None)
    assert qc.CAPTURE.active and c.loaded == 1
    qs = {name: torch.randn(12, 2, 4) for name in LAYERS}
    for name, layer in layers.items():
        layer.forward_impl(qs[name])
    c.wait_for_save()
    saved = np.load(root / "drift-capture" / f"cap-1.rank{rank}.npz")
    assert int(saved["head_offset"]) == rank * 2 and float(saved["scale"]) == pytest.approx(0.0625)
    for index, name in zip((3, 7), LAYERS):
        expected = np.einsum("nhp,hpl->nhl", qs[name][[5, 6, 8]].numpy(), layers[name].W_UK_T.numpy())
        np.testing.assert_allclose(saved[f"l{index}"], expected.astype(np.float16), rtol=1e-3, atol=1e-3)
    assert json.loads((root / "drift-capture" / f"cap-1.rank{rank}.done").read_text())["rows"] == 3
    assert saved["positions"].tolist() == [5, 6, 8]
    assert not qc.CAPTURE.active and not qc.CAPTURE.records


def test_unflagged_steps_record_nothing(connector):
    module, _ = connector
    c = module.DriftGlm53Connector()
    layer = FakeMLA(LAYERS[0])
    qc.install(FakeMLA)
    c.meta = c.build_connector_meta(step("stock"))
    c.start_load_kv(None)
    layer.forward_impl(torch.randn(3, 2, 4))
    assert not qc.CAPTURE.active and not qc.CAPTURE.records and layer.calls == 1


@pytest.mark.parametrize("make,change,reason", [
    (dict(capture=False), {}, "not started with drift_query_capture or drift_hidden_capture"),
    ({}, dict(rows=(5, 20)), "inside the prompt"),
    ({}, dict(rows=(6, 5)), "increasing"),
    ({}, dict(name="../x"), "must match"),
])
def test_bad_requests_are_refused_with_an_error_file(connector, make, change, reason):
    module, root = connector
    c = module.DriftGlm53Connector(**make)
    c.on_new_request(request("r1", **change))
    c.meta = c.build_connector_meta(step("r1"))
    assert reason in c.meta.capture[0].failed
    c.wait_for_save()
    errors = list((root / "drift-capture").glob("*.rank0.error"))
    assert len(errors) == 1 and reason in json.loads(errors[0].read_text())["error"]


def test_a_step_shared_with_another_request_is_refused(connector):
    module, _ = connector
    c = module.DriftGlm53Connector()
    c.on_new_request(request("r1"))
    meta = c.build_connector_meta(step("r1", total=20))
    assert "alone" in meta.capture[0].failed and meta.capture[0].final


def test_rows_split_across_steps_are_gathered_into_one_file(connector):
    module, root = connector
    c = module.DriftGlm53Connector()
    c._drift_rank_failed = bool
    layers = {name: FakeMLA(name, seed=i) for i, name in enumerate(LAYERS)}
    qc.install(FakeMLA)
    c.on_new_request(request("r1"))                                    # rows 5, 6 and 8
    first = c.build_connector_meta(step("r1", scheduled=7))
    assert [(s.positions, s.before, s.final, s.failed) for s in first.capture] == [((5, 6), 0, False, "")]
    later = NS(scheduled_new_reqs=[], scheduled_cached_reqs=NS(req_ids=["r1"], new_block_ids=[None], num_computed_tokens=[7], resumed_req_ids=set()),
               num_scheduled_tokens={"r1": 5}, total_num_scheduled_tokens=5)
    qs = []
    for meta in (first, c.build_connector_meta(later)):
        c.meta = meta
        c.start_load_kv(None)
        q = {name: torch.randn(12, 2, 4) for name in LAYERS}
        qs.append(q)
        for name, layer in layers.items():
            layer.forward_impl(q[name][: 7 if meta is first else 5])
        c.wait_for_save()
        done = root / "drift-capture" / "cap-1.rank0.done"
        assert done.exists() == (meta is not first)
    saved = np.load(root / "drift-capture" / "cap-1.rank0.npz")
    assert saved["positions"].tolist() == [5, 6, 8]
    name = LAYERS[0]
    rows = torch.cat([qs[0][name][[5, 6]], qs[1][name][[1]]])
    expected = np.einsum("nhp,hpl->nhl", rows.numpy(), layers[name].W_UK_T.numpy())
    np.testing.assert_allclose(saved["l3"], expected.astype(np.float16), rtol=1e-3, atol=1e-3)


def test_a_span_in_a_later_step_waits(connector):
    module, _ = connector
    c = module.DriftGlm53Connector()
    c.on_new_request(request("r1", rows=(9, 11)))
    assert c.build_connector_meta(step("r1", scheduled=8)).capture == []
    later = NS(scheduled_new_reqs=[], scheduled_cached_reqs=NS(req_ids=["r1"], new_block_ids=[None], num_computed_tokens=[8], resumed_req_ids=set()),
               num_scheduled_tokens={"r1": 4}, total_num_scheduled_tokens=4)
    ready = c.build_connector_meta(later).capture
    assert [(s.before, s.failed) for s in ready] == [(8, "")]


class FakeKDA:
    def __init__(self, index):
        self.prefix, self.calls = f"language_model.model.layers.{index}.self_attn", 0

    def forward(self, hidden_states, positions):
        self.calls += 1
        return hidden_states


@pytest.fixture
def hidden(connector, monkeypatch):
    from drift.serving import glm_hidden_capture as hc
    hc.CAPTURE.clear()
    yield connector, hc
    hc.CAPTURE.clear()


def hidden_connector(module, rank, query=False):
    c = module.DriftGlm53Connector(capture=query, rank=rank)
    c._hidden_capture, c._state_groups = True, {2: 64}
    c._layer_to_group.update({f"language_model.model.layers.{i}.self_attn": 2 for i in (0, 1)})
    c._drift_rank_failed = bool
    return c


@pytest.mark.parametrize("rank", [0, 1])
def test_rank_zero_records_each_recurrent_layer_input_at_the_flagged_rows(hidden, rank):
    (module, root), hc = hidden
    c = hidden_connector(module, rank)
    assert hc.install(FakeKDA)
    layers = [FakeKDA(0), FakeKDA(1)]
    c.on_new_request(request("r1"))
    c.meta = c.build_connector_meta(step("r1"))
    c.start_load_kv(None)
    inputs = [torch.randn(12, 8) for _ in layers]
    for layer, value in zip(layers, inputs):
        layer.forward(value, None)
    c.wait_for_save()
    path = root / "drift-capture" / "cap-1.hidden.npz"
    assert path.exists() == (rank == 0)
    if rank == 0:
        saved = np.load(path)
        assert saved["positions"].tolist() == [5, 6, 8]
        for index, value in enumerate(inputs):
            np.testing.assert_allclose(saved[f"h{index}"], value[[5, 6, 8]].numpy().astype(np.float16), rtol=1e-3, atol=1e-3)
        assert json.loads((root / "drift-capture" / "cap-1.rank0.done").read_text())["hidden_layers"] == 2
    assert not hc.CAPTURE.active and not hc.CAPTURE.records


def test_a_compiled_server_that_records_nothing_fails_the_capture(hidden):
    (module, root), hc = hidden
    c = hidden_connector(module, 0)
    hc.install(FakeKDA)
    c.on_new_request(request("r1"))
    c.meta = c.build_connector_meta(step("r1"))
    c.start_load_kv(None)
    FakeKDA(0).forward(torch.randn(12, 8), None)                          # layer 1 never ran through the wrapper
    c.wait_for_save()
    error = json.loads((root / "drift-capture" / "cap-1.rank0.error").read_text())["error"]
    assert "run eagerly" in error
