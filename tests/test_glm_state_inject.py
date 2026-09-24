"""Exported recurrent state is copied byte for byte into a live request's running state block, or refused whole."""
from __future__ import annotations
import json
import struct
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from drift.serving.glm53_handoff import MAGIC
from drift.serving.glm_state_inject import write_state
from tests.test_glm_prefix_reuse import connector, live, step

KDA = ["language_model.model.layers.0.self_attn", "language_model.model.layers.1.self_attn"]
SHAPE = [6, 1, 1, 40]


def raw_blob(pages, rank=0, world=2, n_tokens=100, shape=SHAPE, tokens=None, name="own-1"):
    entries, data = [], b""
    for layer, page in pages.items():
        entries.append({"layer": layer, "part": 0, "group": 2, "kind": "state", "blocks": [9], "pages_per_block": 1,
                        "state_tokens": tokens or [n_tokens], "page_bytes": len(page), "registered_shape": shape,
                        "shape": [1, 1, 1, len(page)], "dtype": "int8", "offset": len(data), "nbytes": len(page)})
        data += page
    raw = json.dumps({"format": "glm53-handoff-raw-v1", "handoff_id": name, "n_tokens": n_tokens, "tp_rank": rank,
                      "tp_size": world, "tensors": entries}).encode()
    start = -(-(16 + len(raw)) // 4096) * 4096
    return MAGIC + struct.pack("<Q", len(raw)) + raw + b"\0" * (start - 16 - len(raw)) + data


def blob(folder, pages, rank=0, **change):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"rank{rank}.bin").write_bytes(raw_blob(pages, rank=rank, **change))
    (folder / f"rank{rank}.ready").write_text(json.dumps({"mode": "file", "created": time.time()}))


def fake(tmp_path, rank=0):
    caches = {name: torch.zeros(SHAPE, dtype=torch.int8) for name in KDA}
    caches["language_model.model.layers.3.self_attn.attn"] = torch.zeros(6, 128, 656, dtype=torch.uint8)
    groups = {KDA[0]: 2, KDA[1]: 3, "language_model.model.layers.3.self_attn.attn": 0}
    return NS(_kv_caches=caches, _layer_to_group=groups, _state_groups={2: 64, 3: 64}, _output_path=tmp_path,
              _live_out=tmp_path / "tp-live-out", _page_rows=lambda tensor, blocks: (list(blocks), 1), _tp=lambda: (rank, 2))


def pages(seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.integers(-128, 128, 32, dtype=np.int8).tobytes() for name in KDA}


def live_step(**change):
    values = dict(name="s", after=100, reserve_start=20, reserve=80, blocks=((0,), (1,), (4, 5), (2, 3)), state_blob="own-1")
    return NS(**{**values, **change})


def test_each_page_lands_in_the_running_block_and_nothing_else_changes(tmp_path):
    c, exported = fake(tmp_path), pages()
    blob(tmp_path / "own-1", exported)
    report = write_state(c, live_step())
    for name, block in ((KDA[0], 5), (KDA[1], 3)):                       # slot (100 - 1) // 64 = 1 of each group
        page = c._kv_caches[name][block].reshape(-1)
        assert page[:32].numpy().tobytes() == exported[name] and not page[32:].any()
        assert int(c._kv_caches[name].abs().sum()) == int(page.abs().sum())
    assert report["parts"] == 2 and report["bytes"] == 64 and report["position"] == 100
    assert json.loads((tmp_path / "tp-live-out" / "s" / "state.rank0.json").read_text())["parts"] == 2


@pytest.mark.parametrize("change,kind,match", [
    (dict(step=dict(after=90)), RuntimeError, "ends at the reserve"),
    (dict(blob=dict(world=1)), ValueError, "not this rank"),
    (dict(blob=dict(name="own-2")), ValueError, "not this rank"),
    (dict(drop=KDA[1]), ValueError, "do not match"),
    (dict(blob=dict(shape=[6, 1, 1, 48])), ValueError, "layout differs"),
    (dict(blob=dict(tokens=[64])), ValueError, "prompt-end"),
])
def test_any_mismatch_refuses_before_a_single_byte_is_written(tmp_path, change, kind, match):
    c, exported = fake(tmp_path), pages()
    exported.pop(change.get("drop", None), None)
    blob(tmp_path / "own-1", exported, **change.get("blob", {}))
    with pytest.raises(kind, match=match):
        write_state(c, live_step(**change.get("step", {})))
    assert not any(c._kv_caches[name].any() for name in KDA)


def test_the_connector_writes_the_state_with_the_first_memory_at_the_boundary(connector):
    c, root = connector
    c._kv_caches[KDA[0]] = torch.zeros(SHAPE, dtype=torch.int8)
    c._layer_to_group[KDA[0]] = 1
    c._state_groups = {1: 64}
    exported = {KDA[0]: pages()[KDA[0]]}
    blob(root / "own-1", exported, world=1, n_tokens=64)
    c.on_new_request(live("r", start=20, reserve=44, prompt=300, drift_tap=False, drift_prefill_boundary=64,
                          drift_state_blob="own-1"))
    (root / "tp-live-in" / "s").mkdir(parents=True)
    np.savez(root / "tp-live-in" / "s" / "000000.npz", **{f"l{i}": np.ones((4, 512), np.float32) for i in (3, 7)})
    meta = c.build_connector_meta(step(new=[("r", ([0, 1, 2], [4, 5]), 0)], scheduled={"r": 64}))
    assert meta.live[0].failed == "" and meta.live[0].apply == (0,) and meta.live[0].state_blob == "own-1"
    c.meta = meta
    c.wait_for_save()
    assert c._kv_caches[KDA[0]][4].reshape(-1)[:32].numpy().tobytes() == exported[KDA[0]]
    assert (root / "tp-live-out" / "s" / "state.rank0.json").exists()
    assert (root / "tp-live-out" / "s" / "ack.000000.rank0.json").exists()


def arena_connector(tmp_path, blob_bytes, inode=7, age=0.0, mapped_inode=7):
    c = fake(tmp_path)
    arena = bytearray(8192) + bytearray(blob_bytes) + bytearray(64)
    c._map_arena, c._arena_ttl = (lambda: (tmp_path / "arena", arena, None, mapped_inode)), 120
    (tmp_path / "own-1").mkdir()
    ready = {"mode": "arena", "offset": 8192, "length": len(blob_bytes), "arena_inode": inode, "created": time.time() - age}
    (tmp_path / "own-1" / "rank0.ready").write_text(json.dumps(ready))
    return c


def test_an_export_in_the_daemon_arena_is_read_from_its_lease(tmp_path):
    exported = pages(1)
    c = arena_connector(tmp_path, raw_blob(exported))
    write_state(c, live_step())
    assert c._kv_caches[KDA[0]][5].reshape(-1)[:32].numpy().tobytes() == exported[KDA[0]]


@pytest.mark.parametrize("change,match", [(dict(age=121.0), "lease"), (dict(mapped_inode=8), "not mapped")])
def test_an_expired_or_foreign_arena_is_refused(tmp_path, change, match):
    c = arena_connector(tmp_path, raw_blob(pages(1)), **change)
    with pytest.raises(ValueError, match=match):
        write_state(c, live_step())
    assert not any(c._kv_caches[name].any() for name in KDA)
