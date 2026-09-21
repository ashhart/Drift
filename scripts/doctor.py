"""Read-only environment/config inventory. Does not load drivers or download weights."""
import argparse
import json
from pathlib import Path
from drift.runtime.manifest import inventory

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    parser.error("output exists; choose a new audit file")
args.out.write_text(json.dumps(inventory(args.model), indent=2) + "\n")
print(args.out)
