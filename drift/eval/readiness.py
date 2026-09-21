"""Fail-closed plan inspection, never trial authorization or scientific credit."""

import hashlib
import json

from .readiness_rules import (
    check_budget, check_design, check_failure_policy, check_freeze,
    check_isolation, check_schedule,
)


def inspect_plan(plan):
    errors = []
    if not isinstance(plan, dict):
        errors.append("plan: object required")
    else:
        if plan.get("schema") != "drift.appointment-readiness.v1":
            errors.append("schema: unsupported")
        if plan.get("claim_scope") not in ("exploratory_appointment", "formal_m2"):
            errors.append("claim_scope: explicit supported scope required")
        for check in (check_freeze, check_budget, check_design, check_failure_policy, check_isolation, check_schedule):
            check(plan, errors)
    return {
        "status": "BLOCKED" if errors else "PASSED",
        "scope": "plan_completeness_only",
        "launch_authorized": False,
        "publication_authorized": False,
        "scientific_stage_advanced": False,
        "evidence_verified": False,
        "blockers": errors,
        "required_external_checks": [
            "Verify approval, frozen artifacts and preregistration timestamp before hidden evaluation.",
            "Verify denial tests under actual worker and agent identities, including tools and shared storage.",
            "Qualify the selected schedule and all applicable upstream engineering gates.",
            "Enforce the approved budget and run the scorer outside model and agent access.",
        ],
    }


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("nonfinite JSON number")


def inspect_bytes(raw):
    if len(raw) > 1_048_576:
        raise ValueError("plan exceeds metadata size limit")
    plan = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    result = inspect_plan(plan)
    result["plan_sha256"] = hashlib.sha256(raw).hexdigest()
    return result
