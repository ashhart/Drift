"""Answer-level training triples from stored code contexts, for studio_train_memory_answer.py.

Two sources, each holding the exact text the sender read, so questions are drawn from that text, never from a
repository as it is now:
  --contexts  question files (project_questions.py output) of whole modules; the file stem is the passage id;
  --windows   window lists [{id, passage}] cut from longer files (drift/eval/window_questions.py); the id's prefix
              before "-" is the passage id, as the exporters name their files.
With --merge, an existing triples file keeps its train and val items and the new items join its train set. A passage
that is a validation passage is refused.

  python3 scripts/live/code_triples.py --contexts codetrain/c*.json --kinds recall --per-context 6 \\
      --merge codetrain_triples.json --out coderecall_triples.json
  python3 scripts/live/code_triples.py --windows corpus_windows.json stdlib_windows.json \\
      --kinds signature,recall,raises,continue --per-kind 1 --merge coderecall_triples.json --out window_triples.json
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.eval import code_questions, window_questions

parser = argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--contexts", type=Path, nargs="+", help="question files {context, files, questions}")
source.add_argument("--windows", type=Path, nargs="+", help="window lists [{id, passage}]")
parser.add_argument("--kinds", default="recall")
parser.add_argument("--per-context", type=int, default=6, help="whole modules: at most this many new questions each, sampled")
parser.add_argument("--per-kind", type=int, default=1, help="windows: at most this many questions of each kind per window, sampled")
parser.add_argument("--merge", type=Path, help="existing triples {train, val} to extend")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
rng, added, kinds = random.Random(args.seed), [], args.kinds.split(",")
base = json.loads(args.merge.read_text()) if args.merge else {"train": [], "val": []}
held = {x["pid"] for x in base["val"]}


def keep(pid: str, passage: str, found: list[dict]):
    if pid in held:
        raise SystemExit(f"{pid} is a validation passage")
    for q in found:
        added.append({"pid": pid, "passage": passage, "question": q["question"], "answer": q["answer"], "kind": q["kind"]})


if args.contexts:
    for path in sorted(args.contexts):
        context = json.loads(path.read_text())["context"]
        found = code_questions.questions(code_questions.split_context(context), kinds)
        keep(path.stem, context, rng.sample(found, min(args.per_context, len(found))))
else:
    for path in args.windows:
        for window in json.loads(path.read_text()):
            found = window_questions.questions(window["passage"], kinds)
            chosen = [q for kind in kinds for q in rng.sample([q for q in found if q["kind"] == kind], min(args.per_kind, sum(q["kind"] == kind for q in found)))]
            keep(window["id"].split("-")[0], window["passage"], chosen)
args.out.write_text(json.dumps({"train": base["train"] + added, "val": base["val"]}) + "\n")
print(json.dumps({"added": len(added), "kinds": {k: sum(x["kind"] == k for x in added) for k in kinds},
                  "train": len(base["train"]) + len(added), "val": len(base["val"])}))
