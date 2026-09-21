"""E4 workspace boundary (spec M5.2, D6, D11).

Each member works in its own separate clone (not a worktree: a shared `.git` store
is not private). Hidden tests, scorer outputs and the audit log live under an audit
root that no clone contains. Peer artifacts are exchanged only through the
controlled merge service, which logs every authorized read with its byte count so
the E4 artifact channel is metered separately from the KV channel. E1/E2 runs use
`peer_reads_allowed=False`, which makes any peer read a hard failure.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@dataclass
class Workspace:
    root: Path
    audit_root: Path
    members: dict[str, Path] = field(default_factory=dict)
    starting_commit: str | None = None
    peer_reads_allowed: bool = False
    log_path: Path | None = None

    @classmethod
    def create(cls, source_repo: Path, root: Path, audit_root: Path, members: list[str],
               peer_reads_allowed: bool) -> "Workspace":
        if audit_root.resolve() == root.resolve() or audit_root.resolve() in root.resolve().parents or root.resolve() in audit_root.resolve().parents:
            raise ValueError("the audit root must be disjoint from the workspace root")
        root.mkdir(parents=True, exist_ok=False)
        audit_root.mkdir(parents=True, exist_ok=True)
        start = _git("rev-parse", "HEAD", cwd=source_repo)
        ws = cls(root, audit_root, peer_reads_allowed=peer_reads_allowed, starting_commit=start,
                 log_path=audit_root / "artifact_channel.jsonl")
        for name in members:
            clone = root / name
            subprocess.run(["git", "clone", "--quiet", str(source_repo), str(clone)], check=True, capture_output=True)
            _git("checkout", "--quiet", start, cwd=clone)
            _git("checkout", "--quiet", "-b", f"member/{name}", cwd=clone)
            ws.members[name] = clone
        ws._log({"event": "created", "starting_commit": start, "members": members, "peer_reads_allowed": peer_reads_allowed})
        return ws

    def _log(self, record: dict) -> None:
        record = {"t": time.time(), **record}
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_peer_artifact(self, reader: str, owner: str, relative: str) -> bytes:
        """The only sanctioned way for one member to see another's files."""
        if reader == owner or reader not in self.members or owner not in self.members:
            raise ValueError("invalid reader/owner")
        if not self.peer_reads_allowed:
            self._log({"event": "peer_read_denied", "reader": reader, "owner": owner, "path": relative})
            raise PermissionError("peer artifact reads are forbidden in this arm")
        target = (self.members[owner] / relative).resolve()
        if not target.is_relative_to(self.members[owner].resolve()) or ".git" in target.parts:
            raise PermissionError("path escapes the owner's working tree or touches the git store")
        data = target.read_bytes()
        self._log({"event": "peer_read", "reader": reader, "owner": owner, "path": relative,
                   "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return data

    def commit_member(self, name: str, message: str) -> str:
        clone = self.members[name]
        _git("add", "-A", cwd=clone)
        subprocess.run(["git", "-c", "user.name=drift", "-c", "user.email=drift@local", "commit", "--quiet",
                        "--allow-empty", "-m", message], cwd=clone, check=True, capture_output=True)
        sha = _git("rev-parse", "HEAD", cwd=clone)
        self._log({"event": "commit", "member": name, "sha": sha})
        return sha

    def merge_and_test(self, integration_repo: Path, test_command: list[str], hidden_tests_dir: Path | None = None) -> dict:
        """Controlled merge: fetch every member branch into a separate integration clone,
        merge them onto the starting commit, then run the tests there (with hidden tests
        copied in from the audit root). Results are written to the audit root only."""
        subprocess.run(["git", "clone", "--quiet", str(self.members[next(iter(self.members))]), str(integration_repo)], check=True, capture_output=True)
        _git("checkout", "--quiet", "-B", "integration", self.starting_commit, cwd=integration_repo)
        merged, conflicts = [], []
        for name, clone in self.members.items():
            _git("fetch", "--quiet", str(clone), f"member/{name}:member/{name}", cwd=integration_repo)
            result = subprocess.run(["git", "-c", "user.name=drift", "-c", "user.email=drift@local", "merge", "--quiet",
                                     "--no-edit", f"member/{name}"], cwd=integration_repo, capture_output=True, text=True)
            if result.returncode == 0:
                merged.append(name)
            else:
                conflicts.append(name)
                subprocess.run(["git", "merge", "--abort"], cwd=integration_repo, capture_output=True)
        if hidden_tests_dir is not None:
            if not hidden_tests_dir.resolve().is_relative_to(self.audit_root.resolve()):
                raise ValueError("hidden tests must live under the audit root")
            subprocess.run(["cp", "-R", str(hidden_tests_dir) + "/.", str(integration_repo / "hidden_tests")], check=True)
        run = subprocess.run(test_command, cwd=integration_repo, capture_output=True, text=True)
        report = {"merged": merged, "conflicts": conflicts, "returncode": run.returncode,
                  "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest()}
        (self.audit_root / "merge_result.json").write_text(json.dumps(report, indent=2))
        (self.audit_root / "merge_stdout.txt").write_text(run.stdout + run.stderr)
        self._log({"event": "merge_and_test", **report})
        return report

    def channel_ledger(self) -> dict:
        reads = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        peer = [r for r in reads if r["event"] == "peer_read"]
        return {"peer_reads": len(peer), "peer_bytes": sum(r["bytes"] for r in peer),
                "denied": sum(1 for r in reads if r["event"] == "peer_read_denied")}
