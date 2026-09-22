import json
from pathlib import Path

import pytest

from scripts.deploy_glm.bundle import load_manifest, prepare, validate_sources
from scripts.deploy_glm.check import check_targets

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "configs/glm_connector_deployment.json"


def test_private_hostnames_are_operator_supplied_not_hardcoded(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    for index, target in enumerate(manifest['targets']):
        target.update(ssh=f'compute-{index}.example', container=f'model-rank-{index}')
    path = tmp_path / 'deployment.json'
    path.write_text(json.dumps(manifest))
    assert load_manifest(path)['targets'] == manifest['targets']


@pytest.mark.parametrize('name', ['host;true', '-oProxyCommand=x', '', 'has space'])
def test_deployment_target_names_cannot_inject_commands(tmp_path, name):
    manifest = json.loads(MANIFEST.read_text())
    manifest['targets'][0]['ssh'] = name
    path = tmp_path / 'deployment.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identity'):
        load_manifest(path)


def test_complete_live_bundle_and_local_preparation(tmp_path):
    manifest = load_manifest(MANIFEST)
    validate_sources(ROOT, manifest)
    prepared = prepare(ROOT, manifest, tmp_path / "candidate")
    assert prepared["promotion_performed"] is False
    assert sorted(p.name for p in (tmp_path / "candidate").glob("*.py")) == [
        "drift_glm53_connector.py", "glm53_handoff.py", "glm_allocation_lifecycle.py", "glm_cache_commit.py", "glm_prefill_boundary.py", "glm_prefill_scheduler.py", "glm_rank_failure.py",
        "glm_worker_save.py", "live_publication.py", "live_rank_receipt.py", "live_receiver_glm.py",
        "live_tap_capture.py", "live_tap_finish.py", "live_tap_lock.py",
    ]
    assert (tmp_path / "candidate/manifest.json").exists()
    assert (tmp_path / "candidate/rollback-inventory.json").exists()
    with pytest.raises(FileExistsError):
        prepare(ROOT, manifest, tmp_path / "candidate")


def test_source_hash_change_blocks_preparation_before_any_write(tmp_path):
    manifest = load_manifest(MANIFEST)
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        prepare(ROOT, manifest, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_new_flat_dependency_cannot_bypass_pinned_closure(tmp_path):
    import hashlib
    manifest = load_manifest(MANIFEST)
    serving = tmp_path / "drift/serving"
    serving.mkdir(parents=True)
    for spec in manifest["files"]:
        source = ROOT / spec["source"]
        text = source.read_text()
        if spec["module"] == "live_receiver_glm":
            text += "\nfrom new_receiver_helper import extra\n"
        (tmp_path / spec["source"]).write_text(text)
        spec["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    (serving / "new_receiver_helper.py").write_text("extra = True\n")
    with pytest.raises(ValueError, match="dependency"):
        validate_sources(tmp_path, manifest)
    from scripts.deploy_glm.bundle import pin_sources
    pinned = pin_sources(tmp_path, manifest, "new-base")
    assert len(pinned["files"]) == len(manifest["files"]) + 1
    assert any(s["module"] == "new_receiver_helper" for s in pinned["files"])
    validate_sources(tmp_path, pinned)


def test_default_check_is_read_only_and_both_targets_are_required():
    manifest = load_manifest(MANIFEST)
    calls = []
    def execute(target, mode, candidate, manifest):
        calls.append((target["ssh"], mode, candidate))
        return {"host": target["ssh"], "valid": target["ssh"] == "spark-a.invalid", "imports": False}
    result = check_targets(manifest, execute=execute)
    assert result["verdict"] == "BLOCKED"
    assert result["promotion_performed"] is False
    assert calls == [("spark-a.invalid", "check", None), ("spark-b.invalid", "check", None)]


def test_staged_check_requires_two_successful_import_receipts():
    manifest = load_manifest(MANIFEST)
    def execute(target, mode, candidate, manifest):
        return {"host": target["ssh"], "valid": True, "imports": target["ssh"] == "spark-a.invalid"}
    result = check_targets(manifest, candidate="/tmp/drift-candidate", execute=execute)
    assert result["verdict"] == "BLOCKED"
    assert not result["promotion_performed"]


def test_target_failure_does_not_skip_other_target():
    manifest = load_manifest(MANIFEST)
    seen = []
    def execute(target, mode, candidate, manifest):
        seen.append(target["ssh"])
        raise TimeoutError()
    result = check_targets(manifest, execute=execute)
    assert seen == ["spark-a.invalid", "spark-b.invalid"]
    assert result["verdict"] == "BLOCKED"
    assert all(r["error"] == "TimeoutError" for r in result["targets"])


def test_real_command_builder_uses_only_read_only_ssh_operations(monkeypatch):
    from types import SimpleNamespace
    from scripts.deploy_glm.check import execute_target
    calls = []
    manifest = load_manifest(MANIFEST)
    target = manifest["targets"][0]
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "inspect" in command[-1]:
            return SimpleNamespace(stdout=json.dumps({"image": target["image"], "running": True}))
        return SimpleNamespace(stdout='DRIFT_DEPLOY_CHECK={"host":"spark-a.invalid","valid":false,"imports":false}\n')
    monkeypatch.setattr("scripts.deploy_glm.check.subprocess.run", run)
    result = execute_target(target, "check", None, manifest)
    assert result["valid"] is False
    assert len(calls) == 2
    assert all(command[0] == "ssh" and options["timeout"] == 45 for command, options in calls)
    assert "docker inspect" in calls[0][0][-1]
    assert "docker exec -i" in calls[1][0][-1]
    assert "-B -S -" in calls[1][0][-1]
    assert all(word not in command[-1] for command, _ in calls for word in ["restart", "docker cp", "systemctl", "sudo"])


def test_remote_payload_imports_all_staged_modules_without_bytecode(tmp_path):
    import hashlib
    import subprocess
    import sys
    from scripts.deploy_glm import remote_check
    site, stage = tmp_path / "active", tmp_path / "candidate"
    site.mkdir()
    stage.mkdir()
    base = 'class Glm53HandoffConnector: pass\n'
    (site / "glm53_handoff_connector.py").write_text(base)
    sources = {
        "drift_glm53_connector": 'import glm53_handoff_connector, glm53_handoff, live_receiver_glm, live_publication\nclass DriftGlm53Connector(glm53_handoff_connector.Glm53HandoffConnector): pass\n',
        "glm53_handoff": 'VALUE = 1\n', "live_receiver_glm": 'VALUE = 2\n', "live_publication": 'VALUE = 3\n',
    }
    files = []
    for name, text in sources.items():
        (stage / (name + ".py")).write_text(text)
        files.append({"module": name, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    target = {"ssh": "fixture", "site_packages": str(site), "python": sys.executable,
              "python_version": sys.version.split()[0], "base_module": "glm53_handoff_connector",
              "base_sha256": hashlib.sha256(base.encode()).hexdigest(), "rollback": {k: None for k in sources}}
    request = {"target": target, "candidate": str(stage), "files": files, "entrypoint": "drift_glm53_connector"}
    script = 'import sys\nsys.path.insert(0, ' + repr(str(site)) + ')\nREQUEST = ' + repr(request) + '\n' + Path(remote_check.__file__).read_text()
    result = subprocess.run([sys.executable, "-B", "-S", "-"], input=script, text=True, capture_output=True, check=True, timeout=10)
    receipt = json.loads(result.stdout.removeprefix("DRIFT_DEPLOY_CHECK="))
    assert receipt["valid"] is True and receipt["imports"] is True
    assert not list(tmp_path.rglob("__pycache__"))
    (stage / "live_publication.py").write_text("raise RuntimeError('must never execute')\n")
    result = subprocess.run([sys.executable, "-B", "-S", "-"], input=script, text=True, capture_output=True, check=True, timeout=10)
    receipt = json.loads(result.stdout.removeprefix("DRIFT_DEPLOY_CHECK="))
    assert receipt["valid"] is False and receipt["imports"] is False
    (stage / "live_publication.py").write_text(sources["live_publication"])
    (stage / "torch.py").write_text("raise RuntimeError('must never execute')\n")
    result = subprocess.run([sys.executable, "-B", "-S", "-"], input=script, text=True, capture_output=True, check=True, timeout=10)
    receipt = json.loads(result.stdout.removeprefix("DRIFT_DEPLOY_CHECK="))
    assert receipt["valid"] is False and receipt["imports"] is False


def test_pin_is_explicit_and_still_enforces_dependency_closure():
    from scripts.deploy_glm.bundle import pin_sources
    manifest = load_manifest(MANIFEST)
    manifest["files"][0]["sha256"] = "0" * 64
    pinned = pin_sources(ROOT, manifest, "new-base")
    assert pinned["files"][0]["sha256"] != "0" * 64
    assert manifest["files"][0]["sha256"] == "0" * 64
    validate_sources(ROOT, pinned)


def test_duplicate_host_receipts_cannot_count_as_two_qualified_sparks():
    manifest = load_manifest(MANIFEST)
    def execute(target, mode, candidate, manifest):
        return {"host": "spark-a.invalid", "valid": True, "imports": True}
    assert check_targets(manifest, candidate="/tmp/candidate", execute=execute)["verdict"] == "BLOCKED"


def test_refresh_rollback_requires_observations_for_every_staged_module():
    from scripts.deploy_glm.check import refresh_rollback
    manifest = load_manifest(MANIFEST)
    def execute(target, mode, candidate, manifest):
        return {"host": target["ssh"], "runtime_match": True, "base_sha256": target["base_sha256"],
                "active_files": {s["module"]: None for s in manifest["files"]}}
    assert all(all(value is None for value in t["rollback"].values()) for t in refresh_rollback(manifest, execute=execute)["targets"])
    with pytest.raises(ValueError, match="unverified"):
        refresh_rollback(manifest, execute=lambda *args: {})
