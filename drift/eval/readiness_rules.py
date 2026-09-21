"""Public metadata requirements for an appointment evaluation plan."""

import math
import re


PIN_NAMES = (
    "models", "quantization", "inference", "prompts", "toolchain", "runner",
    "forward_translator", "reverse_translator", "layer_maps", "injection", "scorer",
)
ISOLATION_CHECKS = (
    "worker_a_peer_read_denied", "worker_b_peer_read_denied",
    "workers_scorer_read_denied", "workers_scorer_write_denied",
    "agents_hidden_evidence_read_denied", "scorer_output_not_returned",
    "separate_process_identities", "tool_and_network_allowlist_enforced",
)
ARMS = {"linked", "no_link", "reverse_only", "forward_only"}
FAILURES = ("timeout", "crash", "rejected", "partial_completion")
INVALIDATIONS = ("leakage", "tampering", "corrupt_activation", "stale_session", "schedule_violation")


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None and len(set(value)) > 1


def positive_integer(value):
    return type(value) is int and value > 0


def finite_number(value, low=0, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def section(plan, name, errors):
    value = plan.get(name)
    if not isinstance(value, dict):
        errors.append(name + ": object required")
        return {}
    return value


def require(values, key, predicate, prefix, errors):
    if not predicate(values.get(key)):
        errors.append(prefix + "." + key + ": missing or invalid")


def check_freeze(plan, errors):
    freeze = section(plan, "freeze", errors)
    for key in ("series_rules_sha256", "approval_record_sha256", "hidden_set_commitment_sha256",
                "development_exclusion_sha256", "preregistration_timestamp_receipt_sha256"):
        require(freeze, key, digest, "freeze", errors)
    require(freeze, "source_commit", lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", x) is not None and len(set(x)) > 1, "freeze", errors)
    for key in ("before_hidden_evaluation", "backbones_frozen", "no_hidden_feedback_for_tuning"):
        require(freeze, key, lambda x: x is True, "freeze", errors)
    pins = section(plan, "pins", errors)
    for key in PIN_NAMES:
        require(pins, key, digest, "pins", errors)


def check_budget(plan, errors):
    budget = section(plan, "budget", errors)
    require(budget, "owner_approval_sha256", digest, "budget", errors)
    for key in ("wall_seconds", "local_tokens_total", "wire_bytes_total", "memory_bytes", "storage_bytes"):
        require(budget, key, positive_integer, "budget", errors)
    require(budget, "retries", lambda x: type(x) is int and x >= 0, "budget", errors)
    require(budget, "scope", lambda x: x == "all_arms_all_repeats_setup_and_retries", "budget", errors)
    require(budget, "enforcement_test_sha256", digest, "budget", errors)


def check_design(plan, errors):
    design = section(plan, "design", errors)
    require(design, "independent_scenarios", lambda x: type(x) is int and x >= 2, "design", errors)
    require(design, "repeats_per_arm", lambda x: type(x) is int and x >= 5, "design", errors)
    require(design, "sampling_unit", lambda x: x == "scenario", "design", errors)
    require(design, "precision_or_power_plan_sha256", digest, "design", errors)
    require(design, "primary_outcome", lambda x: x == "round_trip_correct", "design", errors)
    require(design, "minimum_worthwhile_delta", lambda x: finite_number(x, 0, 1), "design", errors)
    require(design, "arm_order", lambda x: x in ("randomized", "counterbalanced"), "design", errors)
    require(design, "order_seed", lambda x: type(x) is int and x >= 0, "design", errors)
    require(design, "uncertainty", lambda x: x == "paired_bootstrap_scenario_means_ci95", "design", errors)
    require(design, "bootstrap_draws", lambda x: type(x) is int and x >= 100, "design", errors)
    require(design, "bootstrap_seed", lambda x: type(x) is int and x >= 0, "design", errors)
    require(design, "matched_conditions_sha256", digest, "design", errors)
    arms = design.get("arms")
    if not isinstance(arms, list) or any(not isinstance(x, str) for x in arms):
        errors.append("design.arms: list required")
    elif len(arms) != len(set(arms)) or not ARMS.issubset(arms):
        errors.append("design.arms: unique linked and three ablation controls required")
    elif design.get("primary_baseline") not in arms or design.get("primary_baseline") == "linked":
        errors.append("design.primary_baseline: declared non-linked arm required")


def check_failure_policy(plan, errors):
    policy = section(plan, "failure_policy", errors)
    for key in FAILURES:
        require(policy, key, lambda x: x == "score_zero_keep_in_denominator", "failure_policy", errors)
    for key in INVALIDATIONS:
        require(policy, key, lambda x: x == "abort_report_invalid_no_score", "failure_policy", errors)
    for key in ("retain_all_attempts", "report_failure_rates", "separate_startup_and_steady_state",
                "count_local_mail_tokens", "retain_delivery_and_application_failures"):
        require(policy, key, lambda x: x is True, "failure_policy", errors)


def check_isolation(plan, errors):
    isolation = section(plan, "isolation", errors)
    require(isolation, "evidence_outside_agent_and_worker_access", lambda x: x is True, "isolation", errors)
    for key in ISOLATION_CHECKS:
        require(isolation, key, digest, "isolation", errors)


def check_schedule(plan, errors):
    schedule = section(plan, "schedule", errors)
    mode = schedule.get("mode")
    if mode not in ("exploratory_async", "prior_epoch"):
        errors.append("schedule.mode: explicit supported mode required")
    require(schedule, "test_evidence_sha256", digest, "schedule", errors)
    if plan.get("claim_scope") == "formal_m2":
        if mode != "prior_epoch":
            errors.append("schedule.mode: formal M2 requires prior-epoch pinned snapshots")
        require(schedule, "all_layers_pinned_before_either_publish", lambda x: x is True, "schedule", errors)
        gates = section(plan, "upstream_gate_evidence", errors)
        for key in ("M-1", "M0", "M1"):
            require(gates, key, digest, "upstream_gate_evidence", errors)
        arms = plan.get("design", {}).get("arms", []) if isinstance(plan.get("design"), dict) else []
        if not isinstance(arms, list) or not {"solo_a", "solo_b", "text_pair"}.issubset(x for x in arms if isinstance(x, str)):
            errors.append("design.arms: formal M2 requires solo_a, solo_b and text_pair")
