"""Score the causal confirmations by their preregistrations, printing aggregates only.

recall: docs/evaluation/CAUSAL_RECALL_CONFIRMATION.md, measure glm_named_only_true_street, bar 21 of 24.
joint:  docs/evaluation/JOINT_CODE_CAUSAL_CONFIRMATION.md, measure passes, bar 12 of 16.
P1 is a one-sided exact sign test of linked against no_link over disagreeing scenarios, p < 0.05; P2 is the bar; P3 is
parity, linked at most one scenario below text. SUPPORTED needs all three, PARTLY P1 alone. Any invalid arm makes the
run INVALID. No prompt, answer or name is printed.
"""
import argparse
import json
from math import comb
from pathlib import Path

KINDS = {"recall": ("glm_named_only_true_street", 21), "joint": ("passes", 12)}
ARMS = ("linked", "no_link", "text")


def score(report: dict, kind: str) -> dict:
    measure, bar = KINDS[kind]
    rows = report["rows"]
    invalid = sum(1 for r in rows for arm in ARMS if not r.get(arm, {}).get("valid", False))
    got = {arm: [bool(r.get(arm, {}).get(measure)) for r in rows] for arm in ARMS}
    wins = sum(a and not b for a, b in zip(got["linked"], got["no_link"]))
    losses = sum(b and not a for a, b in zip(got["linked"], got["no_link"]))
    n = wins + losses
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0
    totals = {arm: sum(v) for arm, v in got.items()}
    p1, p2, p3 = p < 0.05, totals["linked"] >= bar, totals["linked"] >= totals["text"] - 1
    verdict = "INVALID" if invalid else "SUPPORTED" if p1 and p2 and p3 else "PARTLY" if p1 else "NOT SUPPORTED"
    report_out = {"kind": kind, "scenarios": len(rows), "invalid_arms": invalid, **totals, "linked_only": wins, "no_link_only": losses,
                  "one_sided_p": round(p, 6), "P1": p1, "P2": p2, "P3": p3, "text_below_bar": totals["text"] < bar, "verdict": verdict}
    if kind == "recall":
        positions = {}
        for r, hit in zip(rows, got["linked"]):
            if "position" in r:
                a, b = positions.get(r["position"], (0, 0))
                positions[r["position"]] = (a + hit, b + 1)
        report_out["linked_by_position"] = {str(k): list(v) for k, v in sorted(positions.items())}
        report_out["glm_correct"] = {arm: sum(bool(r.get(arm, {}).get("glm_correct")) for r in rows) for arm in ARMS}
        report_out["qwen_correct"] = {arm: sum(bool(r.get(arm, {}).get("qwen_correct")) for r in rows) for arm in ARMS}
    else:
        report_out["recalled"] = {arm: sum(bool(r.get(arm, {}).get("recalled")) for r in rows) for arm in ARMS}
    return report_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(KINDS))
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.report.read_text()), args.kind), indent=1))
