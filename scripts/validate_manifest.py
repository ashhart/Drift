"""Completeness guard, not a replacement for scientific/permission review."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
missing = []

def walk(value, path="run"):
    if value is None or value == "UNRESOLVED":
        missing.append(path)
    elif isinstance(value, dict):
        for key, child in value.items(): walk(child, path + "." + key)
    elif isinstance(value, list):
        for index, child in enumerate(value): walk(child, f"{path}[{index}]")

walk(manifest)
required = {"schema", "run_uuid", "model_a", "model_b", "layer_pairs", "budgets",
            "channel_policy", "evidence_root", "series_rules_hash", "preregistration_hash"}
missing += ["run." + key for key in sorted(required - manifest.keys())]
if not manifest.get("layer_pairs"):
    missing.append("run.layer_pairs(nonempty)")
print(json.dumps({"status": "BLOCKED" if missing else "COMPLETE_NOT_YET_VALIDATED",
                  "missing": missing}, indent=2))
raise SystemExit(2 if missing else 0)
