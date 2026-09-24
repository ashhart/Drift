"""The causal GLM session waits for the first publication and stops its prefill at the reserve's end."""
import ast
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "scripts/mcdma_target/spark_glm_session.py").read_text())


def statements(first, count):
    """`count` of the script's own top-level statements, starting with the one that assigns `first`."""
    names = lambda node: {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    start = next(i for i, node in enumerate(TREE.body) if isinstance(node, ast.Assign) and first in names(node.targets[0]))
    return compile(ast.Module(body=TREE.body[start:start + count], type_ignores=[]), "<actual-spark-session>", "exec")


def wait(tmp_path, causal, publish, seconds=0.2):
    for kind in ("tp-live-in", "tp-live-out"):
        (tmp_path / kind / "s").mkdir(parents=True)
    if publish:
        (tmp_path / "tp-live-in" / "s" / "000000.npz").write_bytes(b"")
    namespace = dict(ROOT=tmp_path, time=time, shutil=shutil, args=SimpleNamespace(causal=causal, causal_wait=seconds, session="s"))
    exec(statements("first_publication", 3), namespace)


def test_a_causal_session_proceeds_once_the_first_publication_is_released(tmp_path):
    wait(tmp_path, causal=True, publish=True)
    assert (tmp_path / "tp-live-in" / "s").exists()


def test_a_causal_session_without_memory_abandons_before_any_request(tmp_path):
    with pytest.raises(SystemExit, match="abandoned before any request"):
        wait(tmp_path, causal=True, publish=False)
    assert not (tmp_path / "tp-live-in" / "s").exists() and not (tmp_path / "tp-live-out" / "s").exists()


def test_a_concurrent_session_does_not_wait(tmp_path):
    wait(tmp_path, causal=False, publish=False, seconds=0)
    assert (tmp_path / "tp-live-in" / "s").exists()


@pytest.mark.parametrize("causal", [True, False])
def test_the_boundary_is_the_reserve_end_only_in_causal_mode(causal):
    namespace = dict(args=SimpleNamespace(model="m", max_new=8, session="s", tap=True, causal=causal), prompt=[1] * 40,
                     reserve=24, head=[0] * 10, uuid=__import__("uuid"))
    exec(statements("body", 2), namespace)
    params = namespace["body"]["kv_transfer_params"]
    assert params.get("drift_prefill_boundary") == (34 if causal else None)
