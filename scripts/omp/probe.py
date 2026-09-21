"""Qualify the installed OMP extension contract with a network-denied fixture."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def probe_environment(root):
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PI_CODING_AGENT_DIR": str(root / "agent"),
        "PI_CONFIG_DIR": os.path.relpath(root, Path.home()),
        "PI_NO_PTY": "1", "TERM": "dumb", "NO_COLOR": "1", "TMPDIR": str(root),
        "DRIFT_CONTRACT_REPORT": str(root / "facts.json"),
    }


def sandbox_policy(root, user_directory):
    quote = lambda path: json.dumps(str(path))
    dependencies = user_directory / ".bun/install/global/node_modules"
    natives = user_directory / ".omp/natives"
    return "\n".join([
        "(version 1)", "(allow default)", "(deny network*)", "(deny file-write*)",
        f"(allow file-write* (subpath {quote(root)}))",
        f"(deny file-read-data (subpath {quote(user_directory)}))",
        f"(allow file-read-data (subpath {quote(dependencies)}))",
        f"(allow file-read-data (subpath {quote(natives)}))",
        f"(allow file-read-data (subpath {quote(root)}))",
    ])


def qualifies(facts):
    return facts == {
        "registered": True, "stream_calls": 2, "tool_calls": 1,
        "tool_results_seen": 1, "tool_events": 1, "other_tools": 0,
    }


def run(executable, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    executable = executable.resolve(strict=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="drift-omp-contract-") as directory:
        root = Path(directory).resolve()
        (root / "agent").mkdir()
        extension = root / "contract_extension.mjs"
        shutil.copyfile(Path(__file__).with_name(extension.name), extension)
        policy = root / "policy.sb"
        policy.write_text(sandbox_policy(root, Path.home()))
        command = [
            "/usr/bin/sandbox-exec", "-f", str(policy), str(executable),
            "--cwd", str(root), "--no-session", "--no-tools", "--no-lsp", "--no-pty",
            "--no-extensions", "--no-skills", "--no-rules", "--no-title", "--no-prewalk",
            "--extension", str(extension), "--model", "drift-contract-fixture/fixture",
            "--thinking", "off", "--print", "--mode", "json", "--max-time", "20",
            "Run the deterministic fixture.",
        ]
        result = subprocess.run(command, env=probe_environment(root), cwd=root, capture_output=True, timeout=timeout)
        facts_file = root / "facts.json"
        facts = json.loads(facts_file.read_text()) if facts_file.exists() else {}
        return {
            "verdict": "PASSED" if result.returncode == 0 and qualifies(facts) else "BLOCKED",
            "exit_code": result.returncode, "facts": facts,
            "seconds": round(time.monotonic() - started, 3),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "sandbox_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "stderr_kind": "empty" if not result.stderr else "permission_denied" if b"EPERM" in result.stderr else "other",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omp", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=40)
    args = parser.parse_args()
    report = run(args.omp, args.timeout)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["verdict"] == "PASSED" else 2)


if __name__ == "__main__":
    main()
