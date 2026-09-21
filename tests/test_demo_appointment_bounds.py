"""The development driver stops on failure and bounds each delegated run."""
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/live/demo_appointment.py"


@pytest.mark.parametrize("flags", [
    ["--scenarios", "0"], ["--wall-seconds", "nan"],
    ["--startup-seconds", "70", "--wall-seconds", "60"],
    ["--conditions", "linked", "linked"],
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
