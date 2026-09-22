"""Re-score the published note-versus-memory evidence so the headline claim is checked on every run.

This verifies the arithmetic and the preregistered decision rule against the recorded model outputs.
It does not re-run any model, and it does not establish that the result generalises.
"""
import json
from pathlib import Path

import pytest

from scripts.score_note_vs_memory import score

EVIDENCE = Path(__file__).parents[1] / "evidence/note-vs-memory"

pytestmark = pytest.mark.skipif(not EVIDENCE.exists(), reason="published evidence directory is absent")


@pytest.fixture(scope="module")
def rescored():
    return score(EVIDENCE)


def test_the_published_report_reproduces_exactly(rescored):
    published = json.loads((EVIDENCE / "report.json").read_text())
    published["budgets"] = {str(key): value for key, value in published["budgets"].items()}
    assert rescored == published


def test_the_headline_numbers_are_what_was_claimed(rescored):
    assert rescored["questions"] == 240
    assert rescored["memory_em"] == 0.975
    assert rescored["text_em"] == 0.9833
    assert rescored["budgets"]["50"]["note_em"] == 0.8083
    assert rescored["budgets"]["25"]["note_em"] == 0.65


def test_both_preregistered_hypotheses_meet_their_own_thresholds(rescored):
    prereg = json.loads((EVIDENCE / "preregistration.json").read_text())
    assert "N1_note50_loses_facts" in prereg["hypotheses"]
    assert "N2_note25_loses_more" in prereg["hypotheses"]
    fifty, twentyfive = rescored["budgets"]["50"], rescored["budgets"]["25"]
    assert fifty["memory_minus_note"] >= 0.15 and fifty["clustered_sign_flip_p"] < 0.05
    assert twentyfive["memory_minus_note"] >= 0.30 and twentyfive["clustered_sign_flip_p"] < 0.05
    assert rescored["verdict"] == "SUPPORTED"


def test_the_evidence_covers_every_question_in_both_budgets():
    rows = json.loads((EVIDENCE / "memory_arm.report.json").read_text())["rows"]
    answers = {json.loads(line)["key"] for line in (EVIDENCE / "answers.jsonl").read_text().splitlines()}
    assert len(rows) == 240
    for budget in (25, 50):
        missing = [row for row in rows if f"{row['passage']}|{row['kind']}|{budget}" not in answers]
        assert not missing, f"budget {budget} is missing {len(missing)} answers"


def test_the_notes_record_their_own_budget_overruns():
    notes = [json.loads(line) for line in (EVIDENCE / "notes.jsonl").read_text().splitlines()]
    assert {note["budget"] for note in notes} == {25, 50}
    # The writer overran its budget on some notes; that is recorded rather than trimmed away.
    assert max(note["words"] for note in notes if note["budget"] == 25) > 25
