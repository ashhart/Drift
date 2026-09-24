"""The joint code runner scores each arm, keeps the detail out of GLM's linked and no_link prompts, and stops on failure."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/live/demo_joint_code.py"


REAL_RUN = subprocess.run


def fake_loop(glm_text, calls, reverse_errors=()):
    def dispatch(command, **options):
        if "scripts/live/mcdma_loop_test.py" not in command:
            return REAL_RUN(command, **options)                     # the scorer's isolated run of GLM's code
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        out.mkdir()
        messages = json.loads(Path(command[command.index("--glm-messages") + 1]).read_text())
        summary = {"glm": {"text": glm_text(command, messages)}, "reverse_publications": 1, "reverse_errors": list(reverse_errors)}
        (out / "report.json").write_text(json.dumps({"summary": summary}))
        return subprocess.CompletedProcess(command, 0, "", "")
    return dispatch


def run(tmp_path, monkeypatch, dispatch, *extra):
    import scripts.live.loop_evidence as loop_evidence
    monkeypatch.setattr(loop_evidence, "admit_loop_report", lambda report, condition: {"valid": True, "reasons": []})
    monkeypatch.setattr(subprocess, "run", dispatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "run"), "--seed", "5", "--scenarios", "2", *extra])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return json.loads((tmp_path / "run" / "report.json").read_text())


def code_for(name):
    return f"Checking my shared memory for the variable name: {name}\n```python\nimport os\n\ndef get_api_key():\n    return os.environ['{name}']\n```"


def test_each_arm_is_scored_and_only_the_text_arm_is_told(tmp_path, monkeypatch):
    from scripts.live.joint_code_task import scenarios
    names = [s["variable"] for s in scenarios(5, 2)]
    calls = []

    def glm_text(command, messages):
        told = any(name in json.dumps(messages) for name in names)
        assert told == command[command.index("--glm-messages") + 1].endswith("text.glm.json")
        n = int(Path(command[command.index("--out") + 1]).name.split(".")[0][1:])
        return code_for(names[n]) if told else code_for("API_KEY")

    report = run(tmp_path, monkeypatch, fake_loop(glm_text, calls))
    assert [r["text"]["passes"] for r in report["rows"]] == [True, True]
    assert [r["no_link"]["passes"] for r in report["rows"]] == [False, False] and [r["linked"]["passes"] for r in report["rows"]] == [False, False]
    assert all(("--no-link" in c) != c[c.index("--out") + 1].endswith("linked") for c in calls)
    assert all("--publish-text" not in c for c in calls) and report["seed"] == 5


def test_a_skipped_publication_stops_the_run(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="publication skipped"):
        run(tmp_path, monkeypatch, fake_loop(lambda c, m: "", [], reverse_errors=[{"skipped": "full"}]), "--arms", "linked")
    assert json.loads((tmp_path / "run" / "report.json").read_text())["rows"][0]["linked"]["valid"] is False


def test_bad_bounds_fail_before_any_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "run"), "--seed", "1", "--scenarios", "49"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("dispatched"))
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert error.value.code == 2 and not (tmp_path / "run").exists()
