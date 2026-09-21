"""Run the level-1 adapter qualification for one family and write registry evidence.

Usage: PYTHONPATH=. .venv-next/bin/python scripts/qualify_adapter.py --family qwen4_exp --registry registry --model-id <id>
Writes registry/<id>/qualification/level1.json with the junit report hash. Never fabricates.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=["qwen4_exp", "glm5_next"], required=True)
parser.add_argument("--registry", type=Path, required=True)
parser.add_argument("--model-id", required=True)
args = parser.parse_args()
out_dir = args.registry / args.model_id / "qualification"
out_dir.mkdir(parents=True, exist_ok=True)
junit = out_dir / "level1-junit.xml"
result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_adapters_next.py", "-k", args.family,
                         "--junitxml", str(junit)], capture_output=True, text=True)
passed = result.returncode == 0
report = {"level": 1, "family": args.family, "passed": passed, "returncode": result.returncode,
          "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest() if junit.exists() else None,
          "summary": result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""}
(out_dir / "level1.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if passed else 1)
