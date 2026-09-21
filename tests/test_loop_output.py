"""Reject reused direct-loop output before any remote setup."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/live/mcdma_loop_test.py"


def invoke(monkeypatch, output, artifact, attempted):
    class NoRemoteSetup:
        def __init__(self, *args):
            attempted.append(True)

        def __enter__(self):
            raise RuntimeError("synthetic setup stop before remote dispatch")

        def __exit__(self, *args):
            pass

    hosts = SimpleNamespace(OMLX_PY="unused", SPARK="unused", SPARK_PEERS=[], STUDIO="unused")
    monkeypatch.setitem(sys.modules, "livelib", hosts)
    monkeypatch.setitem(sys.modules, "loop_processes", SimpleNamespace(LoopProcesses=NoRemoteSetup, SSH_OPTIONS=[]))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--glm-messages", "unused.json", "--qwen-messages", "unused.json",
                                    "--out", str(output), "--reverse-artifact", str(artifact)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize("has_report", [False, True])
def test_existing_output_is_preserved_and_refused_before_remote_setup(tmp_path, monkeypatch, has_report):
    output = tmp_path / "existing"
    output.mkdir()
    report = output / "report.json"
    original = json.dumps({"session": "synthetic-earlier-run", "status": "PASSED"})
    if has_report:
        report.write_text(original)
    artifact = tmp_path / "translator.npz"
    artifact.write_bytes(b"synthetic-not-loaded")
    attempted = []
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, output, artifact, attempted)
    assert attempted == []
    assert output.is_dir()
    assert report.read_text() == original if has_report else not report.exists()


def test_fresh_output_cannot_inherit_an_earlier_report(tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    artifact = tmp_path / "translator.npz"
    artifact.write_bytes(b"synthetic-not-loaded")
    attempted = []
    with pytest.raises(RuntimeError, match="synthetic setup stop"):
        invoke(monkeypatch, output, artifact, attempted)
    assert attempted == [True]
    assert output.is_dir() and not (output / "report.json").exists()
