from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def checkpoint(root: Path, model_type: str) -> Path:
    root.mkdir()
    config = {
        "model_type": model_type,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 3,
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights-" + model_type.encode())
    (root / "tokenizer.json").write_text(json.dumps({"model": model_type}))
    return root


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "drift.cli", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def create_args(tmp_path: Path) -> tuple[list[str], Path]:
    source = checkpoint(tmp_path / "minimax", "minimax_m3")
    target = checkpoint(tmp_path / "nemotron", "nemotron_lighting_3")
    source_lock = tmp_path / "source.lock"
    target_lock = tmp_path / "target.lock"
    source_lock.write_text("torch==2.10.0\n")
    target_lock.write_text("torch==2.10.0\n")
    output = tmp_path / "my-tap"
    args = [
        "tap", "create", "minimax-nemotron",
        "--source-checkpoint", str(source),
        "--source-adapter", "minimax_m3/torch",
        "--source-host", "host-a",
        "--source-runtime-lock", str(source_lock),
        "--target-checkpoint", str(target),
        "--target-adapter", "nemotron_lighting_3/torch",
        "--target-host", "host-b",
        "--target-runtime-lock", str(target_lock),
        "--output", str(output),
    ]
    return args, output


def test_tap_create_discovers_and_hashes_unknown_models(tmp_path):
    args, output = create_args(tmp_path)
    result = run_cli(*args)
    assert result.returncode == 2
    summary = json.loads(result.stdout)
    assert summary["status"] == "BLOCKED"
    assert summary["tap"] == "minimax-nemotron"
    manifest = json.loads((output / "tap.json").read_text())
    assert manifest["schema"] == "drift.tap.v1"
    assert manifest["status"] == "BLOCKED"
    assert manifest["models"]["source"]["model_type"] == "minimax_m3"
    assert manifest["models"]["target"]["model_type"] == "nemotron_lighting_3"
    assert manifest["models"]["source"]["weights_manifest_sha256"]
    assert manifest["models"]["target"]["tokenizer_manifest_sha256"]
    assert set(manifest["blockers"]) == {
        "source adapter minimax_m3/torch is not installed",
        "source real-weight qualification is missing",
        "source translator is missing",
        "target adapter nemotron_lighting_3/torch is not installed",
        "target real-weight qualification is missing",
        "target translator is missing",
        "remote two-host coordination is not qualified",
    }
    evidence = json.loads((output / "inventory" / "source.json").read_text())
    expected = hashlib.sha256((tmp_path / "minimax" / "config.json").read_bytes()).hexdigest()
    assert evidence["config_sha256"] == expected
    assert "checkpoint" not in result.stdout


def test_tap_status_revalidates_artifacts_and_detects_changes(tmp_path):
    args, output = create_args(tmp_path)
    assert run_cli(*args).returncode == 2
    status = run_cli("tap", "status", str(output))
    assert status.returncode == 2
    assert json.loads(status.stdout)["artifacts_intact"] is True
    (output / "inventory" / "source.json").write_text("{}\n")
    changed = run_cli("tap", "status", str(output))
    assert changed.returncode == 3
    report = json.loads(changed.stdout)
    assert report["status"] == "INVALID"
    assert report["artifacts_intact"] is False


def test_tap_status_detects_checkpoint_changes(tmp_path):
    args, output = create_args(tmp_path)
    assert run_cli(*args).returncode == 2
    (tmp_path / "minimax" / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    changed = run_cli("tap", "status", str(output))
    assert changed.returncode == 3
    report = json.loads(changed.stdout)
    assert report["artifacts_intact"] is True
    assert report["inputs_intact"] is False


def test_tap_create_refuses_existing_output(tmp_path):
    args, output = create_args(tmp_path)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    result = run_cli(*args)
    assert result.returncode == 3
    assert marker.read_text() == "keep"
    assert json.loads(result.stderr)["status"] == "INVALID"


def test_adapter_scaffold_is_a_small_entry_point_package(tmp_path):
    output = tmp_path / "adapter"
    result = run_cli(
        "adapter", "scaffold", "minimax_m3/torch",
        "--model-type", "minimax_m3",
        "--output", str(output),
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "PASSED"
    pyproject = (output / "pyproject.toml").read_text()
    adapter = (output / "src" / "drift_adapter_minimax_m3" / "adapter.py").read_text()
    assert '"minimax_m3/torch" = "drift_adapter_minimax_m3.adapter:load"' in pyproject
    assert "BLOCKED: implement the minimax_m3 adapter" in adapter
    assert "#" not in adapter


def saved_translator(root: Path, member: str, fingerprint: str) -> Path:
    from drift.translate.pool import Layout, PoolFormat, Translator, save_translator

    pool = PoolFormat("pool.v1", 1, 4, fingerprint)
    save_translator(Translator(member, Layout("kv_split", 1, 2), {0: 0}, pool), root, {"test": True})
    return root


def qualification(path: Path, checkpoint_path: Path, adapter_id: str) -> Path:
    from drift.taps.inventory import inspect_checkpoint

    lock = path.parent / (path.stem + ".lock")
    lock.write_text("torch==2.10.0\n")
    inventory = inspect_checkpoint(checkpoint_path, lock)
    report = {
        "level": 2,
        "passed": True,
        "config_sha256": inventory["config_sha256"],
        "weights_manifest_sha256": inventory["weights_manifest_sha256"],
        "tokenizer_manifest_sha256": inventory["tokenizer_manifest_sha256"],
        "runtime_lock_sha256": inventory["runtime_lock_sha256"],
        "adapter": adapter_id,
        "model_type": inventory["model_type"],
    }
    path.write_text(json.dumps(report))
    return path


def test_tap_create_emits_a_runnable_manifest_after_all_local_gates(tmp_path):
    source = checkpoint(tmp_path / "source", "qwen4_exp_text")
    target = checkpoint(tmp_path / "target", "glm5_next_text")
    source_qualification = qualification(tmp_path / "source-qualification.json", source, "qwen4_exp/torch")
    target_qualification = qualification(tmp_path / "target-qualification.json", target, "glm5_next/torch")
    source_lock = tmp_path / "source-qualification.lock"
    target_lock = tmp_path / "target-qualification.lock"
    fingerprint = "ab" * 32
    output = tmp_path / "ready"
    result = run_cli(
        "tap", "create", "ready-tap",
        "--source-checkpoint", str(source),
        "--source-adapter", "qwen4_exp/torch",
        "--source-host", "local",
        "--source-runtime-lock", str(source_lock),
        "--source-qualification", str(source_qualification),
        "--source-translator", str(saved_translator(tmp_path / "source-translator", "qwen", fingerprint)),
        "--target-checkpoint", str(target),
        "--target-adapter", "glm5_next/torch",
        "--target-host", "local",
        "--target-runtime-lock", str(target_lock),
        "--target-qualification", str(target_qualification),
        "--target-translator", str(saved_translator(tmp_path / "target-translator", "glm", fingerprint)),
        "--output", str(output),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASSED"
    run = json.loads((output / "run.json").read_text())
    assert [member["name"] for member in run["members"]] == ["qwen", "glm"]
    assert run["registry_root"] == str(output.resolve() / "registry")
    assert run["pool"]["fingerprint"] == fingerprint
    status = run_cli("tap", "status", str(output))
    assert status.returncode == 0
    assert json.loads(status.stdout)["status"] == "PASSED"


def test_tap_create_rejects_a_forged_or_unloaded_conversion(tmp_path):
    source = checkpoint(tmp_path / "source", "minimax_m3")
    target = checkpoint(tmp_path / "target", "glm5_next_text")
    source_lock, target_lock = tmp_path / "source.lock", tmp_path / "target.lock"
    source_lock.write_text("torch==2.10.0\n")
    target_lock.write_text("torch==2.10.0\n")
    from drift.taps.inventory import inspect_checkpoint

    source_inventory = inspect_checkpoint(source, source_lock)
    (tmp_path / "source-q.json").write_text(json.dumps({
        "level": 2, "passed": True,
        "config_sha256": source_inventory["config_sha256"],
        "weights_manifest_sha256": source_inventory["weights_manifest_sha256"],
    }))
    target_qualification = qualification(tmp_path / "target-q.json", target, "glm5_next/torch")
    weights = tmp_path / "fake"
    weights.mkdir()
    blob = weights / "translator.safetensors"
    blob.write_bytes(b"not a safetensor")
    (weights / "translator.json").write_text(json.dumps({
        "format": 1, "member": "left",
        "pool": {"version": "pool.v1", "levels": 1, "width": 4, "fingerprint": "ab" * 32},
        "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
    }))
    fingerprint = "ab" * 32
    result = run_cli(
        "tap", "create", "forged-tap",
        "--source-checkpoint", str(source),
        "--source-adapter", "qwen4_exp/torch",
        "--source-host", "local",
        "--source-runtime-lock", str(source_lock),
        "--source-qualification", str(tmp_path / "source-q.json"),
        "--source-translator", str(weights),
        "--target-checkpoint", str(target),
        "--target-adapter", "glm5_next/torch",
        "--target-host", "local",
        "--target-runtime-lock", str(target_lock),
        "--target-qualification", str(target_qualification),
        "--target-translator", str(saved_translator(tmp_path / "real", "glm", fingerprint)),
        "--output", str(tmp_path / "forged"),
    )
    assert result.returncode == 2, result.stderr
    blockers = set(json.loads((tmp_path / "forged" / "tap.json").read_text())["blockers"])
    assert "source adapter qwen4_exp/torch does not admit model type minimax_m3" in blockers
    assert "source qualification tokenizer_manifest_sha256 does not match the checkpoint" in blockers
    assert "source translator artifact is invalid" in blockers
    assert not (tmp_path / "forged" / "run.json").exists()


def test_tap_status_rejects_a_rewritten_pass(tmp_path):
    args, output = create_args(tmp_path)
    assert run_cli(*args).returncode == 2
    manifest_path = output / "tap.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "PASSED"
    manifest_path.write_text(json.dumps(manifest))
    changed = run_cli("tap", "status", str(output))
    assert changed.returncode == 3
    report = json.loads(changed.stdout)
    assert report["status"] == "INVALID"
    assert report["artifacts_intact"] is True
    assert report["inputs_intact"] is True
