"""Long project contexts for the payoff measurement, each with needle questions spread through it.

Whole Python modules, in sorted path order, go into a context under their headers until it reaches a character target;
each context gets needle questions (drift/eval/code_questions.py, which function fails with a message), chosen from
modules at evenly spaced depths so a joiner must read the whole memory. The texts are code the resident reads; the
needles check that an attached cache is read at all, not how well.

  python3 scripts/live/payoff_contexts.py --root /path/to/python3.12 --chars 115000,230000,455000 --needles 5 --out payoff_contexts.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.eval.code_questions import context_text, questions


def modules(root: Path) -> list[tuple[str, str]]:
    """ASCII, parseable top-level and package modules of 2,000 to 40,000 characters, in path order; no tests."""
    found = []
    for path in sorted(root.rglob("*.py")):
        relative = str(path.relative_to(root))
        if any(part in ("test", "tests", "idlelib", "lib2to3", "turtledemo", "site-packages") for part in path.relative_to(root).parts):
            continue
        text = path.read_text(errors="replace")
        if text.isascii() and 2000 <= len(text) <= 40000:
            found.append((relative, text))
    return found


def build(files: list[tuple[str, str]], chars: int, needles: int) -> dict:
    """The first modules whose context reaches `chars`, and needles from modules at evenly spaced depths."""
    chosen, total = [], 0
    for relative, text in files:
        if total >= chars:
            break
        room = chars - total - len(relative) - 20
        if room < 200:
            break
        if len(text) > room:                                            # the last module is cut at a line boundary
            text = text[:text.rfind("\n", 0, room) + 1]
        chosen.append((relative, text))
        total += len(text) + len(relative) + 20
    whole = [(r, t) for r, t in chosen if (r, t) in files]              # needles come only from modules kept whole
    picked = []
    for k in range(needles):
        start = int(len(whole) * (k + 0.5) / needles)
        for relative, text in whole[start:] + whole[:start]:
            found = [q for q in questions([(relative, text)], ("raises",)) if q not in picked]
            if found:
                picked.append(found[0])
                break
    context = context_text(chosen)
    return {"text": context, "files": [r for r, _ in chosen], "chars": len(context),
            "questions": [{"id": f"n{i}", "question": q["question"], "answer": q["answer"], "file": q["file"]} for i, q in enumerate(picked)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="a folder of Python modules, such as a standard library")
    parser.add_argument("--chars", required=True, help="comma-separated character targets, one context each")
    parser.add_argument("--needles", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = modules(args.root)
    contexts = [{"id": f"p{i + 1}", **build(files, int(c), args.needles)} for i, c in enumerate(args.chars.split(","))]
    args.out.write_text(json.dumps(contexts) + "\n")
    print(json.dumps([{"id": c["id"], "chars": c["chars"], "files": len(c["files"]), "needles": len(c["questions"])} for c in contexts]))
