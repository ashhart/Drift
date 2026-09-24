"""Score a reverse recall run by docs/evaluation/REVERSE_RECALL_CONFIRMATION.md, printing aggregates only.

Reads the run's report.json and prints counts, the paired sign test and the verdict. It prints no prompt, answer or name,
so the owner can share its output from a held-out run without revealing the set.
"""
import argparse
import json
from math import comb
from pathlib import Path

MEASURE = "glm_named_only_true_street"


def score(report: dict) -> dict:
    rows = report["rows"]
    invalid = sum(1 for r in rows for arm in ("linked", "no_link") if not r.get(arm, {}).get("valid", False))
    pairs = [(bool(r["linked"].get(MEASURE)), bool(r["no_link"].get(MEASURE))) for r in rows
             if r.get("linked", {}).get("valid") and r.get("no_link", {}).get("valid")]
    wins, losses = sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)
    n = wins + losses
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0
    linked = sum(a for a, _ in pairs)
    by_position = {}
    for r in rows:
        if r.get("linked", {}).get("valid") and "position" in r:
            got, total = by_position.get(r["position"], (0, 0))
            by_position[r["position"]] = (got + bool(r["linked"].get(MEASURE)), total + 1)
    verdict = "INVALID" if invalid or len(pairs) != len(rows) else (
        "SUPPORTED" if p < 0.05 and linked >= 18 else "PARTLY" if p < 0.05 else "NOT SUPPORTED")
    return {"scenarios": len(rows), "invalid_arms": invalid, "linked": linked, "no_link": sum(b for _, b in pairs),
            "linked_only": wins, "no_link_only": losses, "one_sided_p": round(p, 6),
            "P1": p < 0.05, "P2": linked >= 18, "linked_by_position": {str(k): list(v) for k, v in sorted(by_position.items())},
            "held_out_inputs": bool(report.get("held_out_inputs")), "verdict": verdict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(json.dumps(score(json.loads(parser.parse_args().report.read_text())), indent=1))
