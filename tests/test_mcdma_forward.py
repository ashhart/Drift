import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drift.serving.mcdma_mailbox import ACK, ACK_AT, PAYLOAD_AT, Reader, Writer


ROOT = Path(__file__).resolve().parents[1]
GL = tuple(3 + 4 * i for i in range(11))
QL = tuple(3 + 4 * i for i in range(12))


class Region:
    region_len = 1 << 20

    def __init__(self):
        self.data = bytearray(self.region_len)

    def get(self, length, offset=0):
        return bytes(self.data[offset:offset + length])

    def put(self, data, offset=0):
        self.data[offset:offset + len(data)] = data


def publication(*, session="session-test", tap=0, start=20, stop=22, arrays=None, byte_delta=0):
    tensors = {f"l{layer}": np.ones((stop - start, 512), np.float16) for layer in GL}
    if arrays is not None:
        tensors.update(arrays)
    body = io.BytesIO()
    np.savez(body, start=np.array(start), stop=np.array(stop), **tensors)
    raw = body.getvalue()
    return json.dumps({"session": session, "tap": tap, "bytes": len(raw) + byte_delta,
                       "file_mtime_ns": 1}).encode() + b"\n" + raw


@pytest.fixture
def drain():
    import time
    source = ast.parse((ROOT / "scripts/live/studio_mcdma_loop.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "drain_forward")
    region = Region()
    reader = Reader(region, session=7)
    writer = Writer(region)
    events = []

    def translate(latents):
        events.append("translate")
        rows = len(latents[GL[0]])
        entries = {layer: (np.ones((rows, 2, 256), np.float16), np.ones((rows, 2, 256), np.float16)) for layer in QL}
        keys = {layer: np.ones((rows, 128), np.float16) for layer in QL}
        return entries, keys, np.arange(rows), np.zeros(rows, dtype=int)

    def append(*args, **kwargs):
        events.append("append")
        assert ACK.unpack(region.get(ACK.size, ACK_AT))[-1] != 1

    env = {"np": np, "io": io, "json": json, "time": time, "forward": reader,
           "args": SimpleNamespace(session="session-test", no_forward=False, own_start=100, foreign_row_cap=100, state_at="arrival", save_taps=None),
           "GL": GL, "QL": QL, "INDEX_DIM": 128, "fwd_used": 0,
           "own_positions": [], "generated": [], "rope": None, "append_entries": append,
           "mx": SimpleNamespace(bfloat16=None, eval=lambda _: events.append("gpu_complete")),
           "cache": {layer: SimpleNamespace(keys=None, values=None) for layer in QL},
           "foreign_bank": SimpleNamespace(positions=lambda sources, query: query - 1 - (sources[-1] - sources), remember=lambda *args: None),
           "fwd": SimpleNamespace(read=translate), "state": None, "lm": None, "pending_state": {}, "rows_fix": None}
    for node in source.body:
        if isinstance(node, ast.ImportFrom) and node.module == "drift.serving.mcdma_forward":
            exec(compile(ast.Module(body=[node], type_ignores=[]), "forward_import", "exec"), env)
    helpers = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in ("advance_state", "corrected")]
    exec(compile(ast.Module(body=[*helpers, function], type_ignores=[]), "drain_forward", "exec"), env)
    return SimpleNamespace(call=env["drain_forward"], env=env, writer=writer, region=region, events=events)


def test_ack_follows_append_and_gpu_completion(drain):
    drain.writer.publish(publication())
    result = drain.call()
    assert result[0]["tap"] == 0
    assert drain.events == ["translate", "append", "gpu_complete"]
    assert drain.writer.acknowledged()


def test_gpu_completion_includes_selector_positions_before_ack(drain):
    expected = []
    for cache in drain.env["cache"].values():
        for name in ("keys", "values", "index_keys", "index_position_ids"):
            value = object()
            setattr(cache, name, value)
            expected.append(value)
    def settle(values):
        assert values == expected
        assert ACK.unpack(drain.region.get(ACK.size, ACK_AT))[-1] != 1
    drain.env["mx"].eval = settle
    drain.writer.publish(publication())
    drain.call()
    assert drain.writer.acknowledged()


@pytest.mark.parametrize("kwargs", [{"session": "stale"}, {"tap": 1}, {"byte_delta": 1},
                                    {"arrays": {"l3": np.ones((2, 511), np.float16)}},
                                    {"arrays": {"l3": np.full((2, 512), np.nan, np.float16)}}])
def test_bad_publication_rejected_before_translation_or_ack(drain, kwargs):
    drain.writer.publish(publication(**kwargs))
    with pytest.raises(Exception):
        drain.call()
    assert drain.events == []
    assert ACK.unpack(drain.region.get(ACK.size, ACK_AT))[-1] != 1


def test_append_failure_is_not_acknowledged_and_cannot_resume(drain):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic append failure")
    drain.env["append_entries"] = fail
    drain.writer.publish(publication())
    with pytest.raises(RuntimeError):
        drain.call()
    assert ACK.unpack(drain.region.get(ACK.size, ACK_AT))[-1] != 1
    before = list(drain.events)
    with pytest.raises(RuntimeError):
        drain.call()
    assert drain.events == before


def test_capacity_exhaustion_aborts_instead_of_acknowledging_skip(drain):
    drain.env["args"].foreign_row_cap = 1
    drain.writer.publish(publication())
    with pytest.raises(Exception):
        drain.call()
    assert ACK.unpack(drain.region.get(ACK.size, ACK_AT))[-1] != 1


def test_corrupt_committed_payload_poisoned_not_ignored(drain):
    drain.writer.publish(publication())
    drain.region.data[PAYLOAD_AT + 3] ^= 1
    with pytest.raises(RuntimeError, match="corrupt"):
        drain.call()
    assert drain.events == []
    with pytest.raises(RuntimeError, match="poisoned"):
        drain.call()


def test_control_drop_is_not_cache_application(drain):
    drain.env["args"].no_forward = True
    drain.writer.publish(publication())
    result = drain.call()
    assert result == [{"session": "session-test", "tap": 0, "dropped_by_control": True, "cache_applied": False}]
    assert drain.events == []
    assert drain.writer.acknowledged()


def test_source_position_gap_rejected_after_valid_tap(drain):
    drain.writer.publish(publication())
    drain.call()
    assert drain.writer.acknowledged()
    drain.writer.publish(publication(tap=1, start=23, stop=25))
    before = list(drain.events)
    with pytest.raises(RuntimeError, match="contiguous"):
        drain.call()
    assert drain.events == before


def test_nonfinite_translator_output_never_appended(drain):
    translate = drain.env["fwd"].read
    def bad(latents):
        entries, keys, order, which = translate(latents)
        keys[QL[-1]][0, 0] = np.inf
        return entries, keys, order, which
    drain.env["fwd"].read = bad
    drain.writer.publish(publication())
    with pytest.raises(RuntimeError, match="nonfinite"):
        drain.call()
    assert drain.events == ["translate"]
    assert not drain.writer.acknowledged()


def test_terminal_count_must_match_applied_taps(drain):
    drain.writer.publish(json.dumps({"op": "complete", "session": "session-test", "tap_count": 1, "bytes": 0, "source_start": 20, "source_stop": 22}).encode() + b"\n")
    with pytest.raises(RuntimeError, match="count mismatch"):
        drain.call()
    assert not drain.writer.acknowledged()


def test_terminal_accepted_after_valid_tap(drain):
    drain.writer.publish(publication())
    drain.call()
    assert drain.writer.acknowledged()
    drain.writer.publish(json.dumps({"op": "complete", "session": "session-test", "tap_count": 1, "bytes": 0, "source_start": 20, "source_stop": 22}).encode() + b"\n")
    assert drain.call() == []
    assert drain.env["forward"].finished
    assert drain.writer.acknowledged()


@pytest.mark.parametrize('bounds', [{'source_start': 20, 'source_stop': 23},
                                  {'source_start': 19, 'source_stop': 22},
                                  {'source_start': True, 'source_stop': 22}])
def test_terminal_must_cover_exact_received_source_range(drain, bounds):
    drain.writer.publish(publication())
    drain.call()
    meta = dict(op='complete', session='session-test', tap_count=1, bytes=0, **bounds)
    drain.writer.publish(json.dumps(meta).encode() + b'\n')
    with pytest.raises(RuntimeError, match='completion'):
        drain.call()
    assert not drain.writer.acknowledged()
    assert not drain.env['forward'].finished


def test_legacy_count_only_terminal_is_rejected(drain):
    drain.writer.publish(json.dumps(dict(op='complete', session='session-test', tap_count=0, bytes=0)).encode() + b'\n')
    with pytest.raises(RuntimeError, match='completion fields'):
        drain.call()
    assert not drain.writer.acknowledged()


def test_bridge_terminal_waits_for_last_tap_ack_and_successful_finish(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("forward_bridge", ROOT / "scripts/mcdma_target/spark_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    region = Region()
    reader = Reader(region, session=9)
    folder = tmp_path / "session-test"
    folder.mkdir()
    body = publication().partition(b"\n")[2]
    (folder / "000000.npz").write_bytes(body)
    bridge = module.Bridge(None, None, str(tmp_path / "in"), str(tmp_path), 0, True, region)
    bridge.watching = "session-test"
    bridge.publish_taps()
    assert json.loads(reader.peek().partition(b"\n")[0])["tap"] == 0
    (folder / "finished").write_text(json.dumps({"failed": "", "writes_scheduled": 1, "tap_count": 1, "source_start": 20, "source_stop": 22}))
    bridge.publish_taps()
    assert json.loads(reader.peek().partition(b"\n")[0])["tap"] == 0
    reader.acknowledge()
    bridge.publish_taps()
    payload = reader.peek()
    assert payload is not None
    meta, _, body = payload.partition(b"\n")
    assert json.loads(meta) == {"op": "complete", "session": "session-test", "tap_count": 1, "bytes": 0, "source_start": 20, "source_stop": 22}
    assert body == b""


def test_completion_requires_both_controller_and_forward_marker():
    from drift.serving.mcdma_completion import wait_for_completion
    calls = []
    def drain():
        calls.append(1)
        return [{"tap": len(calls) - 1}]
    taps = wait_for_completion(io.StringIO("peer_done\n"), drain, lambda: len(calls) == 2, timeout_s=0.2)
    assert taps == [{"tap": 0}, {"tap": 1}]


@pytest.mark.parametrize("text", ["", "go\n", "peer_done extra\n"])
def test_completion_rejects_closed_or_wrong_control(text):
    from drift.serving.mcdma_completion import wait_for_completion
    with pytest.raises(RuntimeError, match="control"):
        wait_for_completion(io.StringIO(text), lambda: [], lambda: True, timeout_s=0.2)


def test_completion_deadline_includes_missing_forward_marker():
    from drift.serving.mcdma_completion import wait_for_completion
    with pytest.raises(TimeoutError, match="terminal"):
        wait_for_completion(io.StringIO("peer_done\n"), lambda: [], lambda: False, timeout_s=0.02)


def test_no_link_still_requires_controller_completion():
    from drift.serving.mcdma_completion import wait_for_completion
    assert wait_for_completion(io.StringIO("peer_done\n"), lambda: [], lambda: True, timeout_s=0.2) == []


def test_a_state_translator_advances_each_linear_layer_before_the_ack(drain):
    rows = []

    class Linear:
        def __init__(self, index):
            self.index = index

        def linear_attn(self, inputs, mask=None, cache=None):
            assert mask is None and cache is drain.env["cache"][self.index] and not drain.writer.acknowledged()
            rows.append((self.index, inputs.shape))
            drain.events.append(f"advance{self.index}")

    width = len(GL) * 512
    drain.env["state"] = {"mean": np.zeros(width, np.float32), "basis": np.eye(width, 8, dtype=np.float32),
                          "layers": {0: (np.ones((8, 6), np.float32), np.zeros(6, np.float32), np.ones(6, np.float32)),
                                     1: (np.ones((8, 6), np.float32), np.zeros(6, np.float32), np.ones(6, np.float32))}}
    drain.env["cache"].update({0: [np.zeros(1), np.zeros(1)], 1: [np.zeros(1), np.zeros(1)]})
    drain.env["lm"] = SimpleNamespace(model=SimpleNamespace(layers={0: Linear(0), 1: Linear(1)}))
    drain.env["mx"] = SimpleNamespace(bfloat16="bf16", eval=lambda _: drain.events.append("gpu_complete"),
                                      array=lambda value: SimpleNamespace(astype=lambda dtype: value))
    drain.writer.publish(publication())
    result = drain.call()
    assert drain.events == ["translate", "append", "gpu_complete", "advance0", "advance1", "gpu_complete"]
    assert [index for index, _ in rows] == [0, 1] and rows[0][1][0] == 1 and rows[0][1][2] == 6
    assert result[0]["state_advanced"] is True and drain.writer.acknowledged()


def test_followup_mode_holds_the_translated_inputs_instead_of_advancing(drain):
    width = len(GL) * 512
    drain.env["args"].state_at = "followup"
    drain.env["state"] = {"mean": np.zeros(width, np.float32), "basis": np.eye(width, 8, dtype=np.float32),
                          "layers": {0: (np.ones((8, 6), np.float32), np.zeros(6, np.float32), np.ones(6, np.float32))}}
    drain.env["lm"] = SimpleNamespace(model=SimpleNamespace(layers={0: None}))                 # any advance would fail here
    drain.writer.publish(publication())
    drain.call()
    held = drain.env["pending_state"][0]
    assert len(held) == 1 and held[0].shape == (len(held[0]), 6) and drain.writer.acknowledged()


def test_a_saved_tap_holds_the_applied_latents_before_the_ack(drain, tmp_path):
    drain.env["args"].save_taps = tmp_path / "taps"
    latents = {f"l{layer}": np.full((2, 512), layer, np.float16) for layer in GL}
    drain.writer.publish(publication(arrays=latents))
    drain.call()
    with np.load(tmp_path / "taps" / "000000.npz") as saved:
        assert (int(saved["start"]), int(saved["stop"])) == (20, 22)
        assert all(np.array_equal(saved[f"l{layer}"], latents[f"l{layer}"]) for layer in GL)
    assert drain.writer.acknowledged()


def test_a_rows_correction_is_added_before_the_append(drain):
    appended = []
    drain.env["append_entries"] = lambda cache, entries, *args, **kwargs: appended.append(entries)
    width, rank = len(GL) * 512, 2
    up = np.zeros((rank, len(QL) * 1024), np.float32)
    up[0, 1024 + 512] = 1.0                                              # the second layer's first value channel
    drain.env["rows_fix"] = {"mean": np.zeros(width, np.float32), "basis": np.eye(width, 4, dtype=np.float32),
                             "down": np.eye(4, rank, dtype=np.float32), "up": up}
    drain.writer.publish(publication(arrays={f"l{layer}": np.full((2, 512), 3.0, np.float16) for layer in GL}))
    drain.call()
    keys, values = appended[0][QL[1]]
    assert np.allclose(values[:, 0, 0], 1 + 3.0) and np.allclose(values[:, 0, 1:], 1) and np.allclose(keys, 1)
    assert np.allclose(appended[0][QL[0]][1], 1) and drain.writer.acknowledged()
