"""The QA pull names its handoffd peer by the configured head host, never by a placeholder."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_jobs_name_the_configured_head_as_their_peer():
    tree = ast.parse((ROOT / "scripts/live/qa_v4_mcdma.py").read_text())
    peers = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
             for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant) and key.value == "peer"]
    assert peers and all(isinstance(value, ast.Name) and value.id == "SPARK" for value in peers)
