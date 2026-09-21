"""Collect two bounded container receipts without copying or activating files."""
import json
from pathlib import Path
import shlex
import subprocess

from .bundle import bundle_id


def ssh(target, remote, *, script=None):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", target["ssh"], shlex.join(remote)]
    return subprocess.run(command, input=script, text=True, capture_output=True, check=True, timeout=45).stdout


def execute_target(target, mode, candidate, manifest):
    template = '{"image":{{json .Image}},"running":{{json .State.Running}}}'
    state = json.loads(ssh(target, ["docker", "inspect", "--format", template, target["container"]]))
    if state != {"image": target["image"], "running": True}:
        return {"host": target["ssh"], "valid": False, "imports": False, "error": "ContainerIdentityMismatch"}
    request = {"target": target, "candidate": candidate, "files": manifest["files"], "entrypoint": manifest["entrypoint"]}
    script = "import json\nREQUEST = json.loads(" + repr(json.dumps(request)) + ")\n" + Path(__file__).with_name("remote_check.py").read_text()
    python_flags = ["-B"] if candidate else ["-B", "-S"]
    raw = ssh(target, ["docker", "exec", "-i", target["container"], target["python"], *python_flags, "-"], script=script)
    lines = [line.removeprefix("DRIFT_DEPLOY_CHECK=") for line in raw.splitlines() if line.startswith("DRIFT_DEPLOY_CHECK=")]
    if len(lines) != 1:
        raise ValueError("missing or ambiguous container receipt")
    return json.loads(lines[0])


def check_targets(manifest, *, candidate=None, execute=execute_target):
    if candidate is not None:
        path = Path(candidate)
        if not path.is_absolute() or ".." in path.parts or any(str(path) == t["site_packages"] for t in manifest["targets"]):
            raise ValueError("candidate must be a separate absolute staged directory")
    results = []
    mode = "check-staged" if candidate else "check"
    for target in manifest["targets"]:
        try:
            receipt = execute(target, mode, candidate, manifest)
        except Exception as error:
            receipt = {"host": target["ssh"], "valid": False, "imports": False, "error": type(error).__name__}
        results.append(receipt)
    passed = len(results) == 2 and all(r.get("host") == t["ssh"] and r.get("valid") is True and (not candidate or r.get("imports") is True) for r, t in zip(results, manifest["targets"]))
    return {"verdict": "PASSED" if passed else "BLOCKED", "mode": mode, "bundle_sha256": bundle_id(manifest), "targets": results,
            "promotion_performed": False, "restart_performed": False, "live_qualification": "NOT_RUN"}


def refresh_rollback(manifest, *, execute=execute_target):
    updated = json.loads(json.dumps(manifest))
    expected = {s["module"] for s in manifest["files"]}
    for target in updated["targets"]:
        receipt = execute(target, "check", None, manifest)
        if (receipt.get("host") != target["ssh"] or receipt.get("runtime_match") is not True
                or receipt.get("base_sha256") != target["base_sha256"]
                or set(receipt.get("active_files", {})) != expected):
            raise ValueError("cannot pin unverified rollback inventory")
        target["rollback"] = receipt["active_files"]
    return updated
