"""Allocation lifecycle regressions at the real connector boundary."""
import numpy as np

from test_vllm_glm53_inject import LAYERS, connector, live_request, request, step


def test_resumed_as_new_replaces_pages_before_injection(connector):
    owner, root = connector
    (root / "tp-inject").mkdir()
    np.savez(root / "tp-inject" / "mem-1.npz",
             l3=np.ones((6, 512), np.float32), l7=np.ones((6, 512), np.float32))
    owner.on_new_request(request("r", tokens=6))
    first = step(new=[("r", ([0], [1]), 0)], scheduled={"r": 4})
    assert owner.build_connector_meta(first).inject == []
    resumed = step(new=[("r", ([0], [5]), 0)], scheduled={"r": 6})
    owner.meta = owner.build_connector_meta(resumed)
    owner.wait_for_save()
    for layer in LAYERS:
        assert not owner._kv_caches[layer][[2, 3]].any()
        assert owner._kv_caches[layer][[10, 11]].any()
    assert owner.meta.inject[0].block_ids == ((0,), (5,))


def test_preemption_discards_pending_allocation(connector):
    owner, _ = connector
    owner.on_new_request(request("r", tokens=6))
    owner.build_connector_meta(step(new=[("r", ([0], [1]), 0)], scheduled={"r": 4}))
    preempted = step()
    preempted.preempted_req_ids = {"r"}
    owner.build_connector_meta(preempted)
    assert owner._tp_pending["r"].block_ids == ()


def test_live_resume_replaces_pages_but_requires_fresh_session(connector):
    owner, _ = connector
    owner.on_new_request(live_request("r", 4, prompt=8))
    owner.build_connector_meta(step(new=[("r", ([0], [1, 2]), 0)], scheduled={"r": 4}))
    resumed = step(new=[("r", ([0], [5]), 0)], scheduled={"r": 4})
    result = owner.build_connector_meta(resumed).live[0]
    assert result.blocks == ((0,), (5,))
    assert "fresh session" in result.failed
    assert result.apply == ()


def test_live_preemption_invalidates_pages_before_any_resume(connector):
    owner, _ = connector
    owner.on_new_request(live_request("r", 4, prompt=8))
    owner.build_connector_meta(step(new=[("r", ([0], [1]), 0)], scheduled={"r": 4}))
    preempted = step()
    preempted.preempted_req_ids = {"r"}
    owner.build_connector_meta(preempted)
    assert owner._live["r"]["blocks"] == ()
    assert "preempted" in owner._live["r"]["failed"]
