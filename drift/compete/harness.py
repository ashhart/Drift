"""M6 competition harness (spec §12, P1 M6).

Competitors submit a POLICY, not a model: translator kind/rank/level map, gate
initialization, scheduling mode and mailbox policy, plus a declared training budget.
Frozen invariants they may not touch: backbones, adapter correctness, answer stores,
channel boundaries, the pool format version. A submission is a signed manifest; the
runner verifies the signature and the invariants, fits the policy within its budget
on the public development pair, evaluates on a frozen held-out set with no
hidden-score feedback to the competitor, and reports quality plus costs with
INVALID / BLOCKED results kept, never ranked.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import torch
from drift.core.attention import Gate
from drift.translate.pool import POOL_VERSION, Layout, PoolFormat, Translator, fit_member, fit_pool_format

ALLOWED_KINDS = {"ridge", "mlp"}
ALLOWED_SCHEDULES = {"one_epoch_lag"}
MAX_RANK = 1024


@dataclass(frozen=True)
class Policy:
    translator_kind: str
    translator_rank: int
    pool_width: int
    gate_initial_logit: float
    schedule: str
    mail_ttl: int
    mail_slots: int
    mail_per_epoch_cap: int
    training_budget_rows: int          # aligned rows the competitor may use for fitting

    def check(self, limits: "Limits") -> None:
        if self.translator_kind not in ALLOWED_KINDS or not 1 <= self.translator_rank <= MAX_RANK:
            raise ValueError("translator kind/rank outside the frozen ABI")
        if self.schedule not in ALLOWED_SCHEDULES:
            raise ValueError("schedule outside the frozen ABI")
        if not 1 <= self.pool_width <= limits.max_pool_width:
            raise ValueError("pool width outside limits")
        if min(self.mail_ttl, self.mail_slots, self.mail_per_epoch_cap) <= 0:
            raise ValueError("mailbox policy must be positive")
        if not 2 <= self.training_budget_rows <= limits.max_training_rows:
            raise ValueError("training budget outside limits")


@dataclass(frozen=True)
class Limits:
    max_pool_width: int
    max_training_rows: int
    max_wall_seconds: float


@dataclass(frozen=True)
class Submission:
    competitor: str
    policy: Policy
    manifest_sha256: str
    signature: str

    @staticmethod
    def sign(competitor: str, policy: Policy, key: bytes) -> "Submission":
        manifest = json.dumps({"competitor": competitor, "policy": asdict(policy)}, sort_keys=True).encode()
        digest = hashlib.sha256(manifest).hexdigest()
        return Submission(competitor, policy, digest, hmac.new(key, digest.encode(), "sha256").hexdigest())

    def verify(self, key: bytes) -> bool:
        manifest = json.dumps({"competitor": self.competitor, "policy": asdict(self.policy)}, sort_keys=True).encode()
        digest = hashlib.sha256(manifest).hexdigest()
        return digest == self.manifest_sha256 and hmac.compare_digest(self.signature, hmac.new(key, digest.encode(), "sha256").hexdigest())


@dataclass
class Invariants:
    """Frozen facts recorded before the competition; every run re-checks them."""
    backbone_digests: dict[str, str]
    pool_version: str = POOL_VERSION

    def check(self, adapters: Mapping[str, Any]) -> None:
        for name, adapter in adapters.items():
            if adapter.frozen_digest() != self.backbone_digests[name]:
                raise RuntimeError(f"backbone {name} changed: INVALID")
        if self.pool_version != POOL_VERSION:
            raise RuntimeError("pool format version changed: INVALID")


@dataclass
class RunResult:
    competitor: str
    verdict: str                          # PASSED | FAILED | INVALID | BLOCKED
    quality: float | None
    costs: dict = field(default_factory=dict)
    reason: str = ""


def run_submission(submission: Submission, key: bytes, limits: Limits, invariants: Invariants,
                   adapters: Mapping[str, Any], layouts: Mapping[str, Layout], level_maps: Mapping[str, Mapping[int, int]],
                   dev_rows: Mapping[int, Mapping[str, torch.Tensor]],
                   evaluate: Callable[[Mapping[str, Translator], Mapping[str, Gate]], float],
                   pass_bar: float, audit_dir: Path) -> RunResult:
    """Fit the policy within its budget on dev rows, evaluate on the frozen held-out set
    (inside `evaluate`, which never returns anything to the competitor), and write the
    record under the audit dir."""
    started = time.perf_counter()
    if not submission.verify(key):
        return RunResult(submission.competitor, "INVALID", None, reason="bad signature")
    try:
        submission.policy.check(limits)
    except ValueError as error:
        return RunResult(submission.competitor, "INVALID", None, reason=str(error))
    try:
        invariants.check(adapters)
    except RuntimeError as error:
        return RunResult(submission.competitor, "INVALID", None, reason=str(error))
    policy = submission.policy
    budget_rows = {level: {m: rows[: policy.training_budget_rows] for m, rows in members.items()} for level, members in dev_rows.items()}
    try:
        fmt, pool_rows = fit_pool_format(budget_rows, width=policy.pool_width)
    except ValueError as error:
        return RunResult(submission.competitor, "BLOCKED", None, reason=f"fit: {error}")
    translators, gates = {}, {}
    for name in adapters:
        t = Translator(name, layouts[name], level_maps[name], fmt, policy.translator_kind, policy.translator_rank)
        fit_member(t, {lvl: budget_rows[lvl][name] for lvl in budget_rows}, pool_rows)
        translators[name] = t
        gates[name] = {layer: Gate(policy.gate_initial_logit) for layer in level_maps[name]}
    wall = time.perf_counter() - started
    if wall > limits.max_wall_seconds:
        return RunResult(submission.competitor, "BLOCKED", None, {"wall_seconds": wall}, "wall budget exhausted")
    quality = float(evaluate(translators, {k: v for g in gates.values() for k, v in g.items()}))
    try:
        invariants.check(adapters)
    except RuntimeError as error:
        return RunResult(submission.competitor, "INVALID", quality, reason=str(error))
    result = RunResult(submission.competitor, "PASSED" if quality >= pass_bar else "FAILED", quality,
                       {"wall_seconds": time.perf_counter() - started, "training_rows": policy.training_budget_rows,
                        "translator_parameters": sum(p.numel() for t in translators.values() for p in t.parameters())})
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"{submission.competitor}.json").write_text(json.dumps({**asdict(result), "manifest_sha256": submission.manifest_sha256}, indent=2))
    return result


def scoreboard(results: Sequence[RunResult]) -> list[dict]:
    """Ranks only PASSED/FAILED by quality; INVALID and BLOCKED are listed, never ranked."""
    ranked = sorted((r for r in results if r.verdict in {"PASSED", "FAILED"}), key=lambda r: -(r.quality or 0.0))
    rows = [{"rank": i + 1, **asdict(r)} for i, r in enumerate(ranked)]
    rows += [{"rank": None, **asdict(r)} for r in results if r.verdict in {"INVALID", "BLOCKED"}]
    return rows
