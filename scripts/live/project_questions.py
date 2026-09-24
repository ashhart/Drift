"""Questions about a real codebase whose answers sit literally in its source, for the drop-in test.

Reads Python files with ast, never runs them (drift/eval/code_questions.py). The project context is the files themselves,
each under a header with its path. The default kinds are default, constant, raises and signature; --kinds adds recall
for training. Writes {context, files, questions: [{id, kind, file, question, answer}]}.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.eval.code_questions import ALL_KINDS, KINDS, context_text, questions

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--files", nargs="+", required=True, help="paths relative to --root, in context order")
parser.add_argument("--name", required=True, help="a short name for this context; question ids start with it")
parser.add_argument("--kinds", default=",".join(KINDS), help=f"comma-separated, from {','.join(ALL_KINDS)}")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
files = [(relative, (args.root / relative).read_text()) for relative in args.files]
found = [{"id": f"{args.name}q{i}", **q} for i, q in enumerate(questions(files, args.kinds.split(",")))]
context = context_text(files)
args.out.write_text(json.dumps({"context": context, "files": args.files, "questions": found}, indent=1) + "\n")
print(json.dumps({"name": args.name, "files": len(args.files), "chars": len(context), "questions": len(found),
                  "kinds": {k: sum(q["kind"] == k for q in found) for k in args.kinds.split(",")}}))
