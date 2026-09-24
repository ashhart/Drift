"""The coordinators pass the operator's MCDMA legs to the Studio and refuse a linked run without them."""
import json
from pathlib import Path
import runpy
import shlex
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LINKS = "192.0.2.1/192.0.2.40,198.51.100.1/198.51.100.40"


class StopAtStudio(RuntimeError):
    pass


def run_loop(monkeypatch, tmp_path, links, spawned, *extra):
    class Owned:
        startup = 1

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, *args, **kwargs):
            return ""

        def spawn(self, host, command):
            spawned.append((host, command))
            if host == "studio-test":
                raise StopAtStudio
            return SimpleNamespace(readline=lambda timeout: json.dumps({"rank": 0, "session": "synthetic"}))

    artifact = tmp_path / "translator.npz"
    artifact.write_bytes(b"synthetic-not-loaded")
    hosts = SimpleNamespace(MCDMA_LINKS=links, OMLX_PY="python3", SPARK="spark-a.invalid", SPARK_PEERS=[], STUDIO="studio-test")
    monkeypatch.setitem(sys.modules, "livelib", hosts)
    monkeypatch.setitem(sys.modules, "loop_processes", SimpleNamespace(LoopProcesses=Owned, SSH_OPTIONS=[]))
    script = ROOT / "scripts/live/mcdma_loop_test.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--glm-messages", str(artifact), "--qwen-messages", str(artifact),
                                      "--out", str(tmp_path / "run"), "--reverse-artifact", str(artifact), *extra])
    monkeypatch.setattr(sys, "path", list(sys.path))
    runpy.run_path(str(script), run_name="__main__")


def studio_argument(spawned, name):
    host, command = spawned[-1]
    assert host == "studio-test"
    words = shlex.split(command.split(" 2>")[0])
    return words[words.index(name) + 1]


def test_the_studio_worker_receives_the_operator_links(monkeypatch, tmp_path):
    spawned = []
    with pytest.raises(StopAtStudio):
        run_loop(monkeypatch, tmp_path, LINKS, spawned)
    assert studio_argument(spawned, "--links") == LINKS


@pytest.mark.parametrize("links", ["", "192.0.2.1/192.0.2.40"])
def test_a_linked_run_without_links_stops_before_any_setup(monkeypatch, tmp_path, links, capsys):
    spawned = []
    with pytest.raises(SystemExit):
        run_loop(monkeypatch, tmp_path, links, spawned)
    assert "DRIFT_MCDMA_LINKS" in capsys.readouterr().err
    assert spawned == [] and not (tmp_path / "run").exists()


def test_a_no_link_run_needs_no_links(monkeypatch, tmp_path):
    spawned = []
    with pytest.raises(StopAtStudio):
        run_loop(monkeypatch, tmp_path, "", spawned, "--no-link")
    assert "--no-link" in shlex.split(spawned[-1][1].split(" 2>")[0])


def test_the_reverse_coordinator_refuses_a_run_without_links(monkeypatch, tmp_path, capsys):
    calls = []
    hosts = SimpleNamespace(MCDMA_LINKS="", OMLX_PY="python3", SPARK="spark-a.invalid", SPARK_PEERS=[], STUDIO="studio-test",
                            sh=lambda *cmd: calls.append(cmd))
    monkeypatch.setitem(sys.modules, "livelib", hosts)
    script = ROOT / "scripts/live/reverse_mcdma_test.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--passages", str(tmp_path / "unread.jsonl"), "--out", str(tmp_path / "run")])
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    assert "DRIFT_MCDMA_LINKS" in capsys.readouterr().err
    assert calls == [] and not (tmp_path / "run").exists()
