"""Exercise remote ownership with local synthetic children and no SSH or model execution."""
import ast
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.live.loop_processes import LoopProcess, LoopProcesses
from scripts.mcdma_target.spark_bridge import acquire_region

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/live/remote_lease.py"


def command(source):
    return shlex.join([sys.executable, "-u", "-c", source])


def start(source, *, session="synthetic-a", seconds=5, lock=None):
    spec = dict(command=command(source), session=session, seconds=seconds, grace=0.05, lock=lock)
    return LoopProcess([sys.executable, str(SUPERVISOR), json.dumps(spec)], session,
                       time.monotonic() + seconds + 2, 0.05)


def assert_stopped(pid):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=1)
        if not state.stdout.strip() or state.stdout.strip().startswith("Z"):
            return
        time.sleep(0.01)
    pytest.fail("owned process is still running")


def test_read_timeout_closes_only_owned_session_and_its_descendant():
    neighbor = start("import os,time; print(os.getpid(),flush=True); time.sleep(20)", session="synthetic-b")
    source = "import os,subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(child.pid,flush=True); time.sleep(20)"
    owned = start(source)
    try:
        neighbor_pid, descendant = int(neighbor.readline(2)), int(owned.readline(2))
        with pytest.raises(TimeoutError):
            owned.readline(0.05)
        owned.close()
        assert owned.stopped["reason"] == "controller_disconnected"
        assert_stopped(descendant)
        os.kill(neighbor_pid, 0)
        assert neighbor.child.poll() is None
    finally:
        owned.close()
        neighbor.close()


def test_remote_lease_stops_command_even_when_controller_stays_connected():
    proc = start("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(20)", seconds=0.2)
    try:
        pid = int(proc.readline(2))
        assert json.loads(proc.readline(2))["reason"] == "lease_expired"
        with pytest.raises(RuntimeError, match="did not complete"):
            proc.finish(2)
        assert_stopped(pid)
    finally:
        proc.close()


def test_disconnect_before_go_does_not_start_work(tmp_path):
    marker = tmp_path / "started"
    source = f"import pathlib,sys; print('ready',flush=True); line=sys.stdin.readline(); pathlib.Path({str(marker)!r}).write_text(line) if line.strip() == 'go' else None"
    proc = start(source)
    assert proc.readline(2) == "ready"
    proc.close()
    assert not marker.exists()


@pytest.mark.parametrize("attempt", range(3))
def test_immediate_disconnect_during_supervisor_startup_is_bounded(attempt):
    proc = start("import time; time.sleep(20)")
    proc.close()
    assert proc.stopped["reason"] == "controller_disconnected"


def test_go_and_completion_are_acknowledged_after_remote_group_stops():
    proc = start("import sys; print('ready',flush=True); assert sys.stdin.readline() == 'go\\n'; print('result',flush=True)")
    try:
        assert proc.readline(2) == "ready"
        proc.release()
        assert proc.readline(2) == "result"
        proc.finish(2)
        assert proc.stopped == dict(runner_stopped="synthetic-a", returncode=0, reason="completed")
    finally:
        proc.close()


def test_peer_done_follows_go_and_is_sent_only_once():
    source = "import sys; print('ready',flush=True); assert sys.stdin.readline() == 'go\\n'; assert sys.stdin.readline() == 'peer_done\\n'; print('complete',flush=True)"
    proc = start(source)
    try:
        assert proc.readline(2) == "ready"
        with pytest.raises(RuntimeError, match="completion order"):
            proc.peer_done()
        proc.release()
        proc.peer_done()
        with pytest.raises(RuntimeError, match="completion order"):
            proc.peer_done()
        assert proc.readline(2) == "complete"
        proc.finish(2)
    finally:
        proc.close()


@pytest.mark.parametrize("frames", [[b"go\npeer_done\n"], [b"g", b"o\npe", b"er_done\n"]])
def test_remote_typed_control_handles_coalesced_and_split_frames(frames):
    source = "import sys; print('ready',flush=True); assert sys.stdin.readline() == 'go\\n'; assert sys.stdin.readline() == 'peer_done\\n'"
    proc = start(source)
    try:
        assert proc.readline(2) == "ready"
        for frame in frames:
            proc.child.stdin.write(frame)
            proc.child.stdin.flush()
        proc.finish(2)
    finally:
        proc.close()


@pytest.mark.parametrize("frames", [b"peer_done\n", b"go\ngo\n", b"go\npeer_done\npeer_done\n", b"go\nprivate text", b"go\npeer_done anything\n"])
def test_remote_typed_control_rejects_extra_or_out_of_order_data(frames):
    proc = start("import time; print('ready',flush=True); time.sleep(20)")
    try:
        assert proc.readline(2) == "ready"
        proc.child.stdin.write(frames)
        proc.child.stdin.flush()
        assert json.loads(proc.readline(2))["reason"] == "invalid_control"
    finally:
        proc.close()


def test_partial_startup_failure_closes_already_started_remote_group(monkeypatch):
    from scripts.live import loop_processes
    original = subprocess.Popen
    started = []

    def local_ssh(argv, **kwargs):
        if started:
            raise OSError("synthetic second host startup failure")
        child = original(["/bin/sh", "-c", argv[-1]], **kwargs)
        started.append(child)
        return child

    with LoopProcesses("partial", seconds=5, startup=2, grace=0.05) as owned:
        monkeypatch.setattr(loop_processes.subprocess, "Popen", local_ssh)
        first = owned.spawn("unused.invalid", command("import os,time; print(os.getpid(),flush=True); time.sleep(20)"))
        pid = int(first.readline(2))
        with pytest.raises(OSError, match="second host"):
            owned.spawn("unused.invalid", "unused")
        monkeypatch.setattr(loop_processes.subprocess, "Popen", original)
    assert_stopped(pid)
    assert first.stopped["reason"] == "controller_disconnected"


def test_region_lease_refuses_concurrent_bridge_before_any_region_write(tmp_path):
    path = tmp_path / "region.bin"
    path.write_bytes(b"unchanged synthetic region")
    source = "from scripts.mcdma_target.spark_bridge import acquire_region; import sys; acquire_region(sys.argv[1])"
    with acquire_region(path):
        result = subprocess.run([sys.executable, "-c", source, str(path)], cwd=ROOT,
                                capture_output=True, timeout=3)
        assert result.returncode != 0
        assert b"BlockingIOError" in result.stderr
        assert path.read_bytes() == b"unchanged synthetic region"
    with acquire_region(path):
        assert path.read_bytes() == b"unchanged synthetic region"


def test_supervisor_lock_conflict_never_launches_second_child(tmp_path):
    lock = tmp_path / "lease"
    marker = tmp_path / "second-started"
    first = start("import time; print('ready',flush=True); time.sleep(20)", lock=str(lock))
    second = None
    try:
        assert first.readline(2) == "ready"
        second = start(f"from pathlib import Path; Path({str(marker)!r}).touch()", lock=str(lock))
        assert json.loads(second.readline(2))["reason"] == "startup_failed"
        assert not marker.exists()
        assert first.child.poll() is None
    finally:
        if second:
            second.close()
        first.close()


def test_coordinator_has_no_global_kill_and_studio_guards_precede_mailbox_mutation():
    coordinator = (ROOT / "scripts/live/mcdma_loop_test.py").read_text()
    studio = (ROOT / "scripts/live/studio_mcdma_loop.py").read_text()
    assert "pkill" not in coordinator
    assert ".stdout.readline(" not in coordinator
    assert "subprocess.run(" not in coordinator
    assert coordinator.index("glm.finish(") < coordinator.index("qwen.peer_done()") < coordinator.index("qwen.finish(")
    assert '"tokens_per_s_overall": None' in coordinator
    assert 'str(args.reverse_artifact.resolve())' in coordinator
    assert 'f"{STUDIO}:drift-frontier/local/live/stacked3_rev.npz"' in coordinator
    assert studio.index('acquire("studio_mcdma_loop.py"') < studio.index("forward = Reader(")
    tree = ast.parse(studio)
    go = next(node for node in tree.body if isinstance(node, ast.If) and "sys.stdin.readline" in ast.unparse(node.test))
    assert ast.unparse(go.test) == "sys.stdin.readline().strip() != 'go'"
    assert isinstance(go.body[0], ast.Raise)
