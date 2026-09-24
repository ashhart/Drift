"""The derived DeepSeek compose file adds only the connector mounts and one --kv-transfer-config argument."""
import json
import runpy
from pathlib import Path
import pytest

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/deploy_dsv4/compose_drift.py"))
derive, MODULES = module["derive"], module["MODULES"]
RECIPE = """services:
  vllm-dspark:
    ipc: host
    volumes:
      - ${HF_CACHE}:/cache/huggingface
    command:
      - bash
      - -lc
      - >
        exec /usr/local/bin/vllm serve model
        --enable-prefix-caching
        --enable-chunked-prefill
        --nnodes 2
"""


def test_mounts_and_one_argument_are_added_and_nothing_else_changes():
    out = derive(RECIPE, "/srv/drift")
    body = out.split("\n", 1)[1]
    added = [line for line in body.splitlines() if line not in RECIPE.splitlines()]
    assert len(added) == len(MODULES) + 1
    assert all(f"- /srv/drift/{name}:/usr/local/lib/python3.12/dist-packages/{name}:ro" in body for name in MODULES)
    argument = next(line for line in added if "--kv-transfer-config" in line)
    assert argument.startswith("        --kv-transfer-config '") and json.loads(argument.split("'")[1])["kv_connector"] == "DriftRawRowConnector"
    assert body.index("--enable-chunked-prefill") < body.index("--kv-transfer-config") < body.index("--nnodes 2")
    assert [line for line in body.splitlines() if line not in added] == RECIPE.splitlines()


@pytest.mark.parametrize("recipe", [RECIPE.replace("--enable-chunked-prefill", "--chunked"), RECIPE.replace("    volumes:", "    volume:"),
                                    RECIPE + "        --kv-transfer-config '{}'\n"])
def test_a_changed_recipe_is_refused(recipe):
    with pytest.raises(ValueError):
        derive(recipe, "/srv/drift")


def test_expandable_segments_are_turned_off_and_an_unknown_setting_is_refused():
    recipe = RECIPE.replace("    volumes:", '    environment:\n      PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n    volumes:')
    out = derive(recipe, "/srv/drift")
    assert 'PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"' in out and "expandable_segments:True" not in out
    with pytest.raises(ValueError, match="expandable segments"):
        derive(recipe.replace("--nnodes 2", "--nnodes 2 # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"), "/srv/drift")
