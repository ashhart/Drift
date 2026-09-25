"""Score the own-cache gate: answer hits per arm, and how closely each memory arm reproduces the text arm's output."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.eval.answer_match import matches

ARMS = ("text", "none", "rows", "rows_state", "t_rows", "t_rows_state", "t_state", "d_rows", "d_state", "d_rows_state")


def shared(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def hit(result: dict, arm: str) -> bool:
    """Whether the arm's answer carries the key, by drift.eval.answer_match's rules for each kind of question."""
    return matches(result["answer"], result.get("question", ""), result[arm]["text"])


def score(results: list[dict]) -> dict:
    """Per arm, over the items that ran it; an arm some items skipped also reports its item count."""
    report = {"items": len(results)}
    for arm in ARMS:
        present = [r for r in results if arm in r and "text" in r[arm]]   # a skipped arm records why, not an answer
        if not present:
            continue
        if len(present) < len(results):
            report[f"{arm}_items"] = len(present)
        report[f"{arm}_hits"] = sum(hit(r, arm) for r in present)
        beside = [r for r in present if "text" in r]                    # a run limited by --arms may have no text arm
        if arm not in ("text", "none") and beside:
            report[f"{arm}_identical_to_text"] = sum(r[arm]["text"] == r["text"]["text"] for r in beside)
            report[f"{arm}_shared_prefix_chars_median"] = sorted(shared(r[arm]["text"], r["text"]["text"]) for r in beside)[len(beside) // 2]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print(json.dumps(score(json.loads(parser.parse_args().results.read_text())["results"]), indent=2))
