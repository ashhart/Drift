"""The development driver stops on failure and bounds each delegated run."""
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/live/demo_appointment.py"


@pytest.mark.parametrize("flags", [
    ["--scenarios", "0"], ["--wall-seconds", "nan"],
    ["--startup-seconds", "70", "--wall-seconds", "60"],
    ["--conditions", "linked", "linked"],
    ["--reserve", "10"],
])
def test_bad_bounds_fail_before_any_dispatch(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "run"), *flags])
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration dispatched a process")
    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("timed_out", [False, True])
def test_failed_arm_stops_remaining_arms_without_printing_private_output(tmp_path, monkeypatch, capsys, timed_out):
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(output), "--scenarios", "1",
                                     "--wall-seconds", "90", "--total-seconds", "100", "--summary-only"])
    calls = []
    def dispatch(command, **options):
        calls.append((command, options))
        Path(command[command.index("--out") + 1]).mkdir()
        if timed_out:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return subprocess.CompletedProcess(command, 2, "synthetic private output", "synthetic private error")
    monkeypatch.setattr(subprocess, "run", dispatch)
    with pytest.raises(SystemExit):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert len(calls) == 1
    command, options = calls[0]
    assert 0 < float(command[command.index("--wall-seconds") + 1]) <= 85
    assert options["timeout"] <= 100
    assert "--reverse-artifact" in command
    report = json.loads((output / "report.json").read_text())
    assert report["rows"][0]["linked"]["valid"] is False
    assert "synthetic private" not in capsys.readouterr().out


def scenario_streets(command):
    fact = json.loads(Path(command[command.index("--qwen-messages") + 1]).read_text())[1]["content"]
    listing = json.loads(Path(command[command.index("--glm-messages") + 1]).read_text())[1]["content"]
    true = fact.rsplit(" on ", 1)[1].rstrip(".")
    return true, [street for street in re.findall(r" is on ([^.]+)\.", listing) if street != true]


def fake_loop(glm_text, reverse_errors, calls):
    def dispatch(command, **options):
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        out.mkdir()
        summary = {"glm": {"text": glm_text(command)}, "qwen": {"followup_answer": "unknown"},
                   "reverse_publications": 1, "forward_taps": 0, "reverse_errors": reverse_errors}
        (out / "report.json").write_text(json.dumps({"summary": summary}))
        return subprocess.CompletedProcess(command, 0, "", "")
    return dispatch


def run_demo(base, monkeypatch, extra, dispatch):
    import scripts.live.loop_evidence as loop_evidence
    monkeypatch.setattr(loop_evidence, "admit_loop_report", lambda report, condition: {"valid": True, "reasons": []})
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(base / "run"), "--scenarios", "1", "--conditions", "linked",
                                     "--wall-seconds", "90", "--total-seconds", "100", "--summary-only", *extra])
    monkeypatch.setattr(subprocess, "run", dispatch)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return json.loads((base / "run" / "report.json").read_text())


def test_oracle_selection_passes_the_answer_text_and_all_does_not(tmp_path, monkeypatch):
    for selection, sends_text in (("oracle", True), ("all", False)):
        calls = []
        report = run_demo(tmp_path / selection, monkeypatch, ["--selection", selection], fake_loop(lambda command: "", [], calls))
        command = calls[0]
        assert ("--publish-text" in command) is sends_text
        if sends_text:
            fact = json.loads(Path(command[command.index("--qwen-messages") + 1]).read_text())[1]["content"]
            assert command[command.index("--publish-text") + 1] == fact
        assert report["selection"] == selection


def test_reserve_reaches_both_sides_of_the_loop(tmp_path, monkeypatch):
    calls = []
    report = run_demo(tmp_path, monkeypatch, ["--reserve", "2048"], fake_loop(lambda command: "", [], calls))
    assert calls[0][calls[0].index("--reserve") + 1] == "2048"
    assert report["reserve"] == 2048


def test_a_skipped_publication_is_rejected_instead_of_scored(tmp_path, monkeypatch):
    dispatch = fake_loop(lambda command: "", [{"skipped": "GLM's reserved span is full"}], [])
    with pytest.raises(SystemExit) as error:
        run_demo(tmp_path, monkeypatch, ["--selection", "all"], dispatch)
    assert "REVERSE_PUBLICATION_SKIPPED" in str(error.value)
    row = json.loads((tmp_path / "run" / "report.json").read_text())["rows"][0]
    assert row["linked"] == {"valid": False, "reasons": ["REVERSE_PUBLICATION_SKIPPED"]}


@pytest.mark.parametrize("hedged, lenient, strict", [(False, True, True), (True, True, False)])
def test_strict_street_metric_rejects_an_answer_naming_several_streets(tmp_path, monkeypatch, hedged, lenient, strict):
    def glm_text(command):
        true, others = scenario_streets(command)
        named = f"{true} or {others[0]}" if hedged else true
        return f"Notes. Checking my shared memory for the user's appointment: a clinic on {named}. RECOMMENDATION: unknown"
    linked = run_demo(tmp_path, monkeypatch, [], fake_loop(glm_text, [], []))["rows"][0]["linked"]
    assert linked["glm_recalled_true_street"] is lenient
    assert linked["glm_named_only_true_street"] is strict


def test_balanced_fresh_scenarios_reach_the_prompts_and_the_report(tmp_path, monkeypatch):
    from scripts.live.appointment_scenarios import VOCABULARY
    calls = []
    report = run_demo(tmp_path, monkeypatch, ["--scenarios", "8", "--balance", "--vocabulary", "fresh", "--seed", "11"],
                      fake_loop(lambda command: "", [], calls))
    assert (report["balance"], report["vocabulary"], report["seed"]) == (True, "fresh", 11)
    assert sorted(row["position"] for row in report["rows"]) == [0, 0, 1, 1, 2, 2, 3, 3]
    true, others = scenario_streets(calls[0])
    assert true in VOCABULARY["fresh"][0] and set(others) <= set(VOCABULARY["fresh"][0])


def test_more_than_48_scenarios_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "run"), "--scenarios", "49"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert error.value.code == 2 and not (tmp_path / "run").exists()


def test_held_out_inputs_come_from_files_and_the_seed_is_not_recorded(tmp_path, monkeypatch):
    names = {"streets": ["Aster Row", "Birch Hill", "Cotton Way", "Dunmore Road"], "shops": ["Cup One", "Pour House", "Grindstone", "Latte Lab"],
             "clinics": ["Vale surgery"]}
    (tmp_path / "names.json").write_text(json.dumps(names))
    (tmp_path / "seed.txt").write_text("987654\n")
    calls = []
    report = run_demo(tmp_path, monkeypatch, ["--vocabulary-file", str(tmp_path / "names.json"), "--seed-file", str(tmp_path / "seed.txt")],
                      fake_loop(lambda command: "", [], calls))
    assert report["seed"] is None and report["held_out_inputs"] is True and report["vocabulary"] == "owner file"
    assert scenario_streets(calls[0])[0] in names["streets"] and "987654" not in json.dumps(report)


def test_the_text_arm_tells_glm_directly_unlinked_and_only_linked_is_causal(tmp_path, monkeypatch):
    calls, admitted = [], []
    import scripts.live.loop_evidence as loop_evidence
    monkeypatch.setattr(loop_evidence, "admit_loop_report", lambda report, condition: admitted.append(condition) or {"valid": True, "reasons": []})
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "run"), "--scenarios", "1", "--conditions", "linked", "text",
                                     "--causal", "--wall-seconds", "90", "--total-seconds", "200", "--summary-only"])
    monkeypatch.setattr(subprocess, "run", fake_loop(lambda command: "", [], calls))
    runpy.run_path(str(SCRIPT), run_name="__main__")
    linked, text = calls
    fact = json.loads(Path(linked[linked.index("--qwen-messages") + 1]).read_text())[1]["content"]
    told = json.loads(Path(text[text.index("--glm-messages") + 1]).read_text())[1]["content"]
    plain = json.loads(Path(linked[linked.index("--glm-messages") + 1]).read_text())[1]["content"]
    assert told.startswith(fact) and fact not in plain
    assert "--no-link" in text and "--no-link" not in linked
    assert "--causal" in linked and "--causal" in text                   # the loop driver drops it for unlinked arms
    assert admitted == ["linked", "no_link"]
