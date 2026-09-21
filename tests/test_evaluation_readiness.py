import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drift.eval.readiness import inspect_bytes, inspect_plan
from drift.eval.readiness_rules import ISOLATION_CHECKS, PIN_NAMES


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/evaluation_readiness.template.json"
SYNTHETIC_DIGEST = hashlib.sha256(b"synthetic metadata fixture, not evidence").hexdigest()


@pytest.fixture
def plan():
    value = json.loads(TEMPLATE.read_text())
    value["freeze"].update({key: SYNTHETIC_DIGEST for key in value["freeze"] if key.endswith("sha256")})
    value["freeze"].update(source_commit=SYNTHETIC_DIGEST[:40], before_hidden_evaluation=True,
                           backbones_frozen=True, no_hidden_feedback_for_tuning=True)
    value["pins"] = dict.fromkeys(PIN_NAMES, SYNTHETIC_DIGEST)
    value["budget"].update(owner_approval_sha256=SYNTHETIC_DIGEST, wall_seconds=1,
                           local_tokens_total=1, wire_bytes_total=1, memory_bytes=1,
                           storage_bytes=1, retries=0, enforcement_test_sha256=SYNTHETIC_DIGEST)
    value["design"].update(independent_scenarios=2, repeats_per_arm=5,
                           precision_or_power_plan_sha256=SYNTHETIC_DIGEST,
                           primary_baseline="no_link", minimum_worthwhile_delta=0.1,
                           arm_order="counterbalanced", order_seed=0, bootstrap_draws=100,
                           bootstrap_seed=0, matched_conditions_sha256=SYNTHETIC_DIGEST)
    value["isolation"].update(dict.fromkeys(ISOLATION_CHECKS, SYNTHETIC_DIGEST))
    value["isolation"]["evidence_outside_agent_and_worker_access"] = True
    value["schedule"]["test_evidence_sha256"] = SYNTHETIC_DIGEST
    return value


def test_complete_metadata_never_authorizes_or_verifies_evidence(plan):
    result = inspect_plan(plan)
    assert result["status"] == "PASSED"
    assert result["scope"] == "plan_completeness_only"
    for key in ("launch_authorized", "publication_authorized", "scientific_stage_advanced", "evidence_verified"):
        assert result[key] is False


def test_template_is_blocked():
    result = inspect_bytes(TEMPLATE.read_bytes())
    assert result["status"] == "BLOCKED"
    assert result["plan_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


@pytest.mark.parametrize("section", ["freeze", "pins", "budget", "design", "failure_policy", "isolation", "schedule"])
def test_missing_section_blocks(plan, section):
    del plan[section]
    assert inspect_plan(plan)["status"] == "BLOCKED"


@pytest.mark.parametrize("section,key,value", [
    ("budget", "owner_approval_sha256", None),
    ("budget", "wall_seconds", True),
    ("budget", "retries", -1),
    ("budget", "local_tokens_total", 0),
    ("freeze", "source_commit", "main"),
    ("freeze", "before_hidden_evaluation", "true"),
    ("freeze", "development_exclusion_sha256", "0" * 64),
    ("pins", "models", "latest"),
    ("pins", "injection", None),
    ("design", "independent_scenarios", 1),
    ("design", "repeats_per_arm", 4),
    ("design", "sampling_unit", "question"),
    ("design", "minimum_worthwhile_delta", float("nan")),
    ("design", "minimum_worthwhile_delta", float("inf")),
    ("design", "primary_baseline", "linked"),
    ("design", "arms", ["linked", "no_link"]),
    ("design", "arms", [{"malformed": True}]),
    ("design", "arms", ["linked", "no_link", "reverse_only", "forward_only", "linked"]),
    ("design", "bootstrap_draws", 99),
    ("design", "arm_order", "fixed"),
    ("failure_policy", "timeout", "drop"),
    ("failure_policy", "leakage", "score_zero_keep_in_denominator"),
    ("failure_policy", "retain_delivery_and_application_failures", False),
    ("isolation", "agents_hidden_evidence_read_denied", None),
    ("isolation", "separate_process_identities", True),
    ("schedule", "mode", None),
])
def test_invalid_or_missing_metadata_blocks(plan, section, key, value):
    plan[section][key] = value
    assert inspect_plan(plan)["status"] == "BLOCKED"


def test_formal_m2_cannot_claim_async_is_prior_epoch(plan):
    plan["claim_scope"] = "formal_m2"
    result = inspect_plan(plan)
    assert result["status"] == "BLOCKED"
    assert any("prior-epoch" in error for error in result["blockers"])
    plan["schedule"].update(mode="prior_epoch", all_layers_pinned_before_either_publish=True)
    plan["upstream_gate_evidence"] = dict.fromkeys(("M-1", "M0", "M1"), SYNTHETIC_DIGEST)
    plan["design"]["arms"].extend(["solo_a", "solo_b", "text_pair"])
    result = inspect_plan(plan)
    assert result["status"] == "PASSED"
    assert result["scientific_stage_advanced"] is False


@pytest.mark.parametrize("raw", [b'{"schema":1,"schema":2}', b'{"value":NaN}', b'{"value":Infinity}', b'{', b'\xff', b' ' * 1_048_577])
def test_bad_json_never_silently_overwrites_or_accepts_numbers(raw):
    with pytest.raises((ValueError, UnicodeError)):
        inspect_bytes(raw)


@pytest.mark.parametrize("value", [None, [], 42, "invalid"])
def test_root_must_be_object(value):
    assert inspect_plan(value)["status"] == "BLOCKED"


def test_cli_blocks_template_and_does_not_echo_values(tmp_path):
    result = subprocess.run([sys.executable, "scripts/evaluation_readiness.py", str(TEMPLATE)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["launch_authorized"] is False
    path = tmp_path / "malformed.json"
    path.write_text('{"private_fixture": "do-not-echo",')
    result = subprocess.run([sys.executable, "scripts/evaluation_readiness.py", str(path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "do-not-echo" not in result.stdout + result.stderr
