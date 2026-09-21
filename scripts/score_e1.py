"""External exact-match scorer. Never route this output back to either model."""
import argparse
import json
from pathlib import Path
from drift.eval.metrics import paired_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--answers", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists")
answer_rows = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
answers = {r["id"]: r["answers"] for r in answer_rows}
if len(answers) != len(answer_rows) or any(set(r) != {"id", "answers"} for r in answer_rows):
    parser.error("answer JSONL needs unique id/answers rows")
rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
if len(rows) != len(answers) or {r["id"] for r in rows} != set(answers):
    parser.error("missing, duplicate or extra trials; reconcile failures explicitly")
if len({r["id"] for r in rows}) != len(rows):
    parser.error("duplicate predictions")
score = lambda text, accepted: float(text.strip().casefold() in {s.strip().casefold() for s in accepted})
active = [score(r["arms"]["foreign"]["text"], answers[r["id"]]) for r in rows]
floor = [score(r["arms"]["floor"]["text"], answers[r["id"]]) for r in rows]
summary = paired_bootstrap(active, floor)
summary["status"] = "DIAGNOSTIC_ONLY_NOT_FORMAL_E1_OR_E2"
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=2) + "\n")
print(args.out)
