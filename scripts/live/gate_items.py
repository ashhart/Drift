"""Gate items in the drop-in's framing, from question files written by project_questions.py.

Each item carries its context as the passage, with Qwen's chat framing around the memory: the drop-in's system prompt
and the line "Project files, from our shared memory:" before it, the question after it. The passage id is the question
file's name, as the sender's exports are named. Items are chosen by kind, and with --limit a seeded sample is kept.

  python3 scripts/live/gate_items.py --questions h1.json h2.json --kinds raises,default,constant --out heldout_qa_items.json
  python3 scripts/live/gate_items.py --questions h*.json --kinds signature --limit 25 --out heldout_signature_items.json
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

SYSTEM = ("You are linked to another AI model through a shared memory that fills while you work. What your partner knows "
          "and writes arrives in that memory, never in this chat. Use it as your own recollection.")
FRAME = "Project files, from our shared memory:"
HEAD = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{FRAME}\n"


def tail(question: str) -> str:
    return f"\n\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def items(paths: list[Path], kinds: set[str], limit: int | None = None, seed: int = 0) -> list[dict]:
    """Gate items of the given kinds from question files, in file then question order; a seeded sample with a limit."""
    found = []
    for path in paths:
        data = json.loads(path.read_text())
        for q in data["questions"]:
            if q["kind"] in kinds:
                found.append({"id": f"{path.stem}-{q['id']}", "kind": q["kind"], "passage": data["context"], "question": q["question"],
                              "answer": q["answer"], "head_text": HEAD, "tail_text": tail(q["question"])})
    if limit is not None and len(found) > limit:
        keep = set(random.Random(seed).sample(range(len(found)), limit))
        found = [item for i, item in enumerate(found) if i in keep]
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--kinds", required=True, help="comma-separated question kinds to keep")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chosen = items(args.questions, set(args.kinds.split(",")), args.limit, args.seed)
    args.out.write_text(json.dumps(chosen, indent=1) + "\n")
    print(json.dumps({"items": len(chosen), "contexts": len({i["id"].split("-")[0] for i in chosen}),
                      "kinds": {k: sum(i["kind"] == k for i in chosen) for k in args.kinds.split(",")}}))
