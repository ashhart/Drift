"""Re-score the preregistered note-versus-memory evaluation from the published evidence."""
import argparse
import json
from pathlib import Path

from drift.eval.paired_stats import clustered_bootstrap, passage_differences, sign_flip_p

BUDGETS = ((50, 0.15), (25, 0.30))


def contains(gold, text):
    return gold.lower() in (text or "").lower()


def score(directory):
    rows = json.loads((directory / "memory_arm.report.json").read_text())["rows"]
    answers = {json.loads(line)["key"]: json.loads(line)["text"]
               for line in (directory / "answers.jsonl").read_text().splitlines()}
    notes = [json.loads(line) for line in (directory / "notes.jsonl").read_text().splitlines()]
    report = {"kind": "PREREGISTERED EVALUATION (note versus memory)", "questions": len(rows),
              "memory_em": round(sum(contains(row["answer"], row["drift_v4"]) for row in rows) / len(rows), 4),
              "text_em": round(sum(contains(row["answer"], row["text_in_prompt"]) for row in rows) / len(rows), 4),
              "budgets": {}}
    for budget, margin in BUDGETS:
        outcomes, note_hits, in_note = {}, 0, 0
        words = [note["words"] for note in notes if note["budget"] == budget]
        note_text = {note["id"]: note["note"] for note in notes if note["budget"] == budget}
        for row in rows:
            answered = contains(row["answer"], answers[f"{row['passage']}|{row['kind']}|{budget}"])
            note_hits += answered
            in_note += contains(row["answer"], note_text[row["passage"]])
            outcomes.setdefault(row["passage"], []).append((contains(row["answer"], row["drift_v4"]), answered))
        exact = note_hits / len(rows)
        gap = report["memory_em"] - exact
        p = sign_flip_p(list(passage_differences(outcomes).values()))
        report["budgets"][str(budget)] = {
            "note_em": round(exact, 4), "answer_literally_present_in_note": round(in_note / len(rows), 4),
            "memory_minus_note": round(gap, 4), "clustered_sign_flip_p": p,
            "bootstrap_95": clustered_bootstrap(outcomes), "median_note_words": sorted(words)[len(words) // 2],
            "max_note_words": max(words), "supported": gap >= margin and p < 0.05}
    supported = [report["budgets"][str(budget)]["supported"] for budget, _ in BUDGETS]
    report["verdict"] = "SUPPORTED" if all(supported) else "PARTLY" if supported[1] else "NOT SUPPORTED"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/note-vs-memory"))
    parser.add_argument("--compare", action="store_true", help="exit nonzero unless the published report reproduces")
    arguments = parser.parse_args()
    report = score(arguments.evidence)
    print(json.dumps(report, indent=1))
    if arguments.compare:
        published = json.loads((arguments.evidence / "report.json").read_text())
        published["budgets"] = {str(key): value for key, value in published["budgets"].items()}
        if published != report:
            raise SystemExit("re-scored report does not match the published one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
