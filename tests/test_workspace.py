"""E4 workspace boundary: separate clones, metered peer reads, hidden tests outside clones."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest
from drift.eval.workspace import Workspace


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_app.py").write_text("from app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "start"], cwd=repo, check=True)
    return repo


def test_clones_are_separate_and_hidden_tests_stay_outside(source, tmp_path):
    ws = Workspace.create(source, tmp_path / "ws", tmp_path / "audit", ["A", "B"], peer_reads_allowed=True)
    assert (ws.members["A"] / ".git").is_dir() and (ws.members["B"] / ".git").is_dir()
    assert ws.members["A"] / ".git" != ws.members["B"] / ".git"
    hidden = tmp_path / "audit" / "hidden"
    hidden.mkdir()
    (hidden / "test_hidden.py").write_text("from app import add, mul\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
    for clone in ws.members.values():
        assert not list(clone.rglob("test_hidden.py"))
    (ws.members["A"] / "app.py").write_text("def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n")
    ws.commit_member("A", "add mul")
    (ws.members["B"] / "README.md").write_text("docs\n")
    ws.commit_member("B", "docs")
    report = ws.merge_and_test(tmp_path / "integration", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."], hidden)
    assert report["merged"] == ["A", "B"] and report["conflicts"] == [] and report["returncode"] == 0
    assert (tmp_path / "audit" / "merge_result.json").exists()
    with pytest.raises(ValueError):
        ws.merge_and_test(tmp_path / "integration2", ["true"], tmp_path / "ws" / "A")


def test_peer_reads_are_metered_or_denied(source, tmp_path):
    ws = Workspace.create(source, tmp_path / "ws", tmp_path / "audit", ["A", "B"], peer_reads_allowed=True)
    data = ws.read_peer_artifact("B", "A", "app.py")
    assert data.startswith(b"def add")
    with pytest.raises(PermissionError):
        ws.read_peer_artifact("B", "A", "../B/app.py")
    with pytest.raises(PermissionError):
        ws.read_peer_artifact("B", "A", ".git/HEAD")
    ledger = ws.channel_ledger()
    assert ledger == {"peer_reads": 1, "peer_bytes": len(data), "denied": 0}
    strict = Workspace.create(source, tmp_path / "ws2", tmp_path / "audit2", ["A", "B"], peer_reads_allowed=False)
    with pytest.raises(PermissionError):
        strict.read_peer_artifact("B", "A", "app.py")
    assert strict.channel_ledger()["denied"] == 1
    entries = [json.loads(l) for l in (tmp_path / "audit2" / "artifact_channel.jsonl").read_text().splitlines()]
    assert entries[-1]["event"] == "peer_read_denied"
    with pytest.raises(ValueError):
        Workspace.create(source, tmp_path / "ws3", tmp_path / "ws3" / "audit", ["A", "B"], peer_reads_allowed=False)
