"""Keep the published demo runnable: it is the first thing a new reader executes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(*arguments):
    return subprocess.run([sys.executable, "scripts/demo_local_link.py", *arguments],
                          cwd=ROOT, capture_output=True, text=True)


def test_the_demo_runs_and_reports_a_live_channel_with_clean_controls():
    result = run("--json")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["channel_live"] is True and summary["controls_clean"] is True
    assert summary["hard_off_reproduced_the_floor_exactly"] == summary["units"]
    assert summary["foreign_changed_the_output"] > 0


def test_the_table_states_what_the_demo_does_not_establish():
    result = run()
    assert result.returncode == 0, result.stderr
    assert "none of this is a language result" in result.stdout
    assert "evidence/note-vs-memory" in result.stdout


def test_one_unit_is_refused_because_the_leakage_control_needs_a_second():
    result = run("--units", "1")
    assert result.returncode != 0 and "at least two units" in result.stderr
