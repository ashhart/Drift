"""Exercise the live CLIs without hosts, models, or private experiment data."""
import io
import hashlib
import shlex
import json
import runpy
import subprocess
import sys
import threading
import types
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("linked,fault", [(linked, fault) for linked in (False, True)
                        for fault in (None, "glm_exit", "glm_truncated", "tap_corrupt", "deadline", "no_progress", "reserve_full", "transfer_fail", "glm_repeat", "qwen_repeat", "tap_gap", "tap_count", "tap_extra", "qwen_count", "qwen_nan", "forward_overflow", "reverse_nan", "tap_sequence_gap", "tap_worker_error", "background_blocked", "pending_overflow", "peer_error", "peer_missing", "peer_ack_missing", "peer_ack_wrong", "late_pending")
                        if linked or fault not in {"tap_corrupt", "reserve_full", "transfer_fail", "tap_gap", "tap_count", "tap_extra", "qwen_count", "qwen_nan", "forward_overflow", "reverse_nan", "tap_sequence_gap", "tap_worker_error", "pending_overflow", "peer_error", "peer_missing", "peer_ack_missing", "peer_ack_wrong", "late_pending"}])
def test_live_loop_no_link_transfers_no_activations(tmp_path, monkeypatch, linked, fault):
    commands, worker_ops, loads, launches = [], [], [], []
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"system": "Keep your own note.", "glm_user": "A", "qwen_user": "B"}))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["drift_loop.py", "--scenario", str(scenario), "--out", str(out)]
                        + ([] if linked else ["--no-link"])
                        + (["--max-seconds", "0.1"] if fault == "deadline" else [])
                        + (["--reserve", "1"] if fault == "reserve_full" else [])
                        + (["--io-timeout", "0.05"] if fault == "peer_ack_missing" else []))
    from drift.serving import live_control
    queues = []
    original_queue = live_control.PendingMemory
    def capture_queue(*args, **kwargs):
        queue = original_queue(*args, **kwargs)
        queues.append(queue)
        return queue
    monkeypatch.setattr(live_control, "PendingMemory", capture_queue)
    killed = threading.Event()
    worker_blocked = threading.Event()
    blocked_timed_out = threading.Event()

    class Reader:
        def __init__(self, reverse=False):
            self.reverse = reverse
            self.head_dim, self.kv_heads = 2, 0 if reverse else 1

        def read(self, latents, gain):
            return {3: np.full((1, 2), np.nan if fault == "reverse_nan" else 1, np.float32)} if self.reverse else {
                3: (np.full((1, 1, 2), 1e10 if fault == "forward_overflow" else 1, np.float32), np.ones((1, 1, 2), np.float32))}

    def load(*args, **kwargs):
        loads.append("forward")
        return Reader()

    lib = types.ModuleType("livelib")
    lib.GLM_LAYERS = lib.QWEN_LAYERS = (3,)
    lib.OMLX_PY, lib.SPARK, lib.STUDIO, lib.SPARK_PEERS = "python", "spark", "studio", ["peer"]
    lib.load_reader = load
    monkeypatch.setitem(sys.modules, "livelib", lib)
    from drift.translate.stacked import StackedReader

    def reverse(*args, **kwargs):
        loads.append("reverse")
        return Reader(True)

    monkeypatch.setattr(StackedReader, "load", reverse)

    class Channel:
        def __init__(self):
            self.replies = [json.dumps({"ready": True}) + "\n"]

        def write(self, line):
            cmd = json.loads(line)
            worker_ops.append(cmd)
            op = cmd["op"]
            reply = {"ok": True}
            if op == "extend":
                if fault in {"background_blocked", "pending_overflow"}:
                    self.replies.append(None)
                    return
                reply.update(system_tokens=0)
            elif op == "generate":
                if fault == "late_pending":
                    queues[0].publish("synthetic-late-publication", 1, lambda: None)
                if fault == "deadline":
                    self.replies.append(None)
                    return
                reply.update(tokens=0 if fault == "no_progress" else 1,
                             text="repeat this same phrase " * 20 if fault == "qwen_repeat" else "reader reply",
                             done=fault != "no_progress")
            elif op == "tap":
                reply.update(tapped=1 if cmd["first"] == 0 else 0, next_first=2 if fault == "qwen_count" else 1)
            self.replies.append(json.dumps(reply) + "\n")

        def readline(self):
            reply = self.replies.pop(0)
            if reply is None:
                worker_blocked.set()
                if not killed.wait(2):
                    blocked_timed_out.set()
                    raise AssertionError("background failure failed to stop a blocked worker")
                return ""
            return reply

        def flush(self):
            pass

        def close(self):
            pass

    def popen(cmd, **kwargs):
        commands.append(cmd)
        launches.append((cmd, kwargs))
        if "studio_drift_worker.py" in cmd[-1]:
            channel = Channel()
            return types.SimpleNamespace(stdin=channel, stdout=channel, wait=lambda **kw: 0,
                                         poll=lambda: None, terminate=killed.set)
        events = [{"started": True, "own_prompt_tokens": 1, "span_start": 0},
                  {"text": "repeat this same phrase " * 20 if fault == "glm_repeat" else "writer reply"}]
        if fault != "glm_truncated":
            events.append({"done": True})
        def wait(**kw):
            if fault == "background_blocked":
                assert worker_blocked.wait(2)
                return 7
            return 7 if fault == "glm_exit" else 0

        return types.SimpleNamespace(stdout=io.StringIO("".join(json.dumps(e) + "\n" for e in events)),
                                     wait=wait, poll=lambda: 0, terminate=lambda: None)

    receipt = {"sha256": "", "rows": 3}

    def run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "scp" and "to_glm_" in " ".join(cmd):
            receipt["sha256"] = hashlib.sha256(Path(cmd[-2]).read_bytes()).hexdigest()
        if cmd[0] == "ssh" and "rank_delivery_status" in cmd[-1]:
            request = json.loads(shlex.split(cmd[-1])[-1])
            peer = request["rank"] == 1
            ack = {"sequence": request["sequence"], "rank": request["rank"], "world_size": 2, **receipt}
            if peer and fault == "peer_ack_wrong":
                ack["rows"] += 1
            status = {"error": peer and fault == "peer_error", "sha256": receipt["sha256"],
                      "present": not (peer and fault == "peer_missing"),
                      "ack": None if peer and fault == "peer_ack_missing" else ack}
            return types.SimpleNamespace(stdout=json.dumps(status), returncode=0)
        if cmd[0] == "rsync" and "tp-live-out" in cmd[-2]:
            if fault == "transfer_fail":
                raise subprocess.CalledProcessError(23, cmd)
            folder = Path(cmd[-1])
            folder.mkdir(exist_ok=True)
            if fault == "tap_corrupt":
                (folder / "000000.npz").write_bytes(b"broken activation file")
            else:
                np.savez(folder / "000000.npz", l3=np.ones((1, 2)),
                         start=(1 if fault == "reserve_full" else 1024) + int(fault == "tap_gap"),
                         stop=(1 if fault == "reserve_full" else 1024) + (2 if fault == "tap_count" else 1),
                         **({"unexpected": np.array([42])} if fault == "tap_extra" else {}))
                if fault == "pending_overflow":
                    for sequence in range(1, 20):
                        np.savez(folder / f"{sequence:06d}.npz", l3=np.ones((1, 2)), start=1024 + sequence, stop=1025 + sequence)
                if fault == "tap_sequence_gap":
                    np.savez(folder / "000002.npz", l3=np.ones((1, 2)), start=1025, stop=1026)
                if fault == "tap_worker_error":
                    (folder / "error.rank0").write_text("synthetic private error payload")
        elif cmd[0] == "rsync" and "/out/" in cmd[-2]:
            np.savez(cmd[-1], k3=np.full((1, 1, 2), np.nan if fault == "qwen_nan" else 1, dtype=np.float32), v3=np.ones((1, 1, 2)))
        return types.SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    if fault:
        with pytest.raises(RuntimeError, match="INVALID"):
            runpy.run_path(str(ROOT / "scripts/live/drift_loop.py"), run_name="__main__")
        assert not (out / "report.json").exists(), "a broken run must not produce a completed report"
        failure = json.loads((out / "failure.json").read_text())
        assert failure["verdict"] == "INVALID"
        assert "synthetic private error payload" not in json.dumps(failure)
        if fault in {"peer_error", "peer_missing"}:
            assert not any(cmd[0] == "ssh" and cmd[-1].startswith("mv /dev/shm") for cmd in commands)
        if fault in {"peer_ack_missing", "peer_ack_wrong"}:
            assert not any(cmd["op"] == "generate" for cmd in worker_ops)
        if fault == "pending_overflow":
            assert killed.is_set()
            assert not blocked_timed_out.is_set()
            assert failure["failure"]["component"] == "tap"
            assert failure["failure"]["error_type"] == "PendingMemoryFull"
            assert len([cmd for cmd in commands if cmd[0] == "scp" and "to_qwen_" in " ".join(cmd)]) == 8
        if fault == "background_blocked":
            assert killed.is_set()
            assert not blocked_timed_out.is_set()
            assert failure["failure"]["component"] == "glm"
        if fault == "deadline":
            assert failure["failure"]["component"] == "deadline"
        if fault in {"glm_repeat", "qwen_repeat"}:
            source = fault.split("_")[0]
            assert failure["failure"]["error_type"] == "RepetitionDetected"
            assert failure["repetition"][source]["triggered"]
            assert failure["repetition"][source]["reason"] == "repeated_word_ngrams"
            assert "repeat this same phrase" not in json.dumps(failure)
        if fault in {"tap_gap", "tap_count", "tap_extra", "forward_overflow"}:
            assert not any(cmd[0] == "scp" and "to_qwen_" in " ".join(cmd) for cmd in commands)
        if fault in {"qwen_count", "qwen_nan", "reverse_nan"}:
            assert not any(cmd[0] == "scp" and "to_glm_" in " ".join(cmd) for cmd in commands)
        assert any(cmd["op"] == "quit" for cmd in worker_ops)
        return
    runpy.run_path(str(ROOT / "scripts/live/drift_loop.py"), run_name="__main__")
    report = json.loads((out / "report.json").read_text())
    assert report["glm_text"] == "writer reply" and report["qwen_text"] == "reader reply"
    for source in ("glm", "qwen"):
        assert not report["repetition"][source]["triggered"]
        assert report["repetition"][source]["words_seen"] == 2
    transfers = [cmd for cmd in commands if cmd[0] in {"scp", "rsync"}
                 and (".npz" in " ".join(cmd) or "tp-live-out" in " ".join(cmd))]
    session_command = next(cmd[-1] for cmd in commands if "spark_live_session.py --session" in cmd[-1])
    assert "--control-stdin" in session_command
    assert all(options.get("stdin") == subprocess.PIPE for _, options in launches)
    assert any(cmd[0] == "scp" and "drift/serving/live_session_control.py" in cmd for cmd in commands)
    if linked:
        assert transfers and loads == ["forward", "reverse"]
        assert {e["kind"] for e in report["timeline"] if e["who"] == "link"} >= {"glm->qwen", "qwen->glm"}
        assert "--no-link" not in session_command
    else:
        assert not any("rank_delivery_status" in cmd[-1] for cmd in commands)
        assert transfers == [], "no-link still transfers the partner's activation files"
        assert loads == [], "no-link must run without fitted translator artifacts"
        assert not any(c["op"] in {"tap", "append"} for c in worker_ops)
        assert not any(e["who"] == "link" for e in report["timeline"])
        assert "--no-link" in session_command


@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("framed", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_spark_control_keeps_prompt_but_disables_live_connector(tmp_path, monkeypatch, capsys, linked, framed, complete):
    import shutil
    requests, filesystem = [], []
    messages = tmp_path / "messages.json"
    messages.write_text('[{"role":"user","content":"private writer note"}]')
    monkeypatch.setenv("DRIFT_GLM_KEY", "synthetic-test-key")
    monkeypatch.setattr(sys, "argv", ["spark_live_session.py", "--session", "test-session", "--messages", str(messages),
                                     "--reserve", "4"] + ([] if linked else ["--no-link"]))

    def post(req, **kwargs):
        body = json.loads(req.data)
        requests.append(body)
        if req.full_url.endswith("/tokenize"):
            tokenized = {"tokens": [42, 43, 44], "token_strs": ["own", "@@DRIFT@@", "note"]} if framed else {
                "tokens": [42, 43], "token_strs": ["own", "note"]}
            return io.BytesIO(json.dumps(tokenized).encode())
        return io.BytesIO(b'data: {"choices":[{"text":"done"}]}\n\n' + (b'data: [DONE]\n' if complete else b""))

    monkeypatch.setattr(urllib.request, "urlopen", post)
    monkeypatch.setattr(shutil, "rmtree", lambda path, **kw: filesystem.append(str(path)))
    monkeypatch.setattr(Path, "mkdir", lambda path, **kw: filesystem.append(str(path)))
    if complete:
        runpy.run_path(str(ROOT / "scripts/live/spark_live_session.py"), run_name="__main__")
    else:
        with pytest.raises(RuntimeError, match="completion marker"):
            runpy.run_path(str(ROOT / "scripts/live/spark_live_session.py"), run_name="__main__")
        assert not any(json.loads(line).get("done") for line in capsys.readouterr().out.splitlines())
    body = requests[-1]
    assert body["prompt"] == ([42] + [198] * 4 + [44] if framed else [198] * 4 + [42, 43])
    assert body["max_tokens"] == 300 and body["temperature"] == 0
    if linked:
        assert body["kv_transfer_params"]["drift_session"] == "test-session"
        assert body["kv_transfer_params"]["drift_reserve_start"] == int(framed)
        assert filesystem
    else:
        assert "kv_transfer_params" not in body
        assert filesystem == [], "no-link must not open an inbox for peer writes"
