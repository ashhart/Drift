"""Fatal cache failures must reach the engine and poison the request."""
from types import SimpleNamespace as NS

import pytest

from test_vllm_glm53_inject import connector, live_request, step


def test_live_worker_failure_reaches_engine_even_if_marker_write_fails(connector, monkeypatch):
    owner, root = connector
    owner.meta = NS(inject=[], live=[NS(name="s", failed="", reserve=4, reserve_start=0)], requests=[])
    def fail(_):
        raise ValueError("private synthetic payload")
    monkeypatch.setattr(owner, "_live_step", fail)
    owner._live_out = root / "not-a-directory"
    owner._live_out.write_text("occupied")
    with pytest.raises(RuntimeError, match="poisoned") as error:
        owner.wait_for_save()
    assert "private synthetic payload" not in str(error.value)
    assert owner._live_worker["s"]["poisoned"]


def test_scheduler_observes_a_rank_error_before_scheduling_more_writes(connector):
    owner, root = connector
    owner.on_new_request(live_request("r", 4, prompt=8, name="s"))
    folder = root / "tp-live-in/s"
    folder.mkdir(parents=True)
    (folder / "000000.npz").write_bytes(b"not yet admitted")
    out = root / "tp-live-out/s"
    out.mkdir(parents=True)
    (out / "error.rank1").write_text("failed")
    metadata = owner.build_connector_meta(step(new=[("r", ([0], [1]), 0)], scheduled={"r": 4}))
    assert metadata.live[0].failed
    assert metadata.live[0].apply == ()
    owner.meta = metadata
    with pytest.raises(RuntimeError, match="poisoned"):
        owner.wait_for_save()


def test_successful_local_rank_still_fails_when_peer_failed(connector):
    owner, root = connector
    owner._drift_rank_failed = lambda failed: True
    owner._live_step = lambda _: None
    owner.meta = NS(inject=[], live=[NS(name="peer-failed", failed="")], requests=[])
    with pytest.raises(RuntimeError, match="poisoned"):
        owner.wait_for_save()
    assert owner._live_worker["peer-failed"]["poisoned"]
    assert (root / "tp-live-out/peer-failed/error.rank0").exists()
