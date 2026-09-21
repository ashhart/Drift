import importlib.util
import pytest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "scripts/omp/probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("omp_contract_probe", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_environment_never_inherits_provider_secrets(tmp_path):
    probe = load_probe()
    env = probe.probe_environment(tmp_path)
    assert set(env) == {"PATH", "PI_CONFIG_DIR", "PI_CODING_AGENT_DIR", "PI_NO_PTY", "TERM", "NO_COLOR", "TMPDIR", "DRIFT_CONTRACT_REPORT"}
    assert env["PI_CODING_AGENT_DIR"] == str(tmp_path / "agent")
    assert (Path.home() / env["PI_CONFIG_DIR"]).resolve() == tmp_path.resolve()


def test_provider_success_requires_a_real_tool_roundtrip():
    probe = load_probe()
    complete = {"registered": True, "stream_calls": 2, "tool_calls": 1, "tool_results_seen": 1, "tool_events": 1, "other_tools": 0}
    assert probe.qualifies(complete)
    for key in complete:
        broken = {**complete, key: 1 if key == "other_tools" else 0}
        assert not probe.qualifies(broken)


def test_probe_uses_host_sandbox_with_network_and_home_read_denied(tmp_path):
    probe = load_probe()
    policy = probe.sandbox_policy(tmp_path, Path("/Users/example"))
    assert "(deny network*)" in policy
    assert '(deny file-read-data (subpath "/Users/example"))' in policy
    assert "(deny file-write*)" in policy


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_probe_rejects_unbounded_deadlines(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        load_probe().run(tmp_path / "absent", timeout)
