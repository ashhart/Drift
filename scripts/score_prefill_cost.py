"""Score configs/preregistration.prefill-cost.json from a run's rows.jsonl."""
import argparse
import json
import statistics
from pathlib import Path

from drift.eval.paired_stats import clustered_bootstrap, passage_differences, sign_flip_p

MEMORY, TEXT = "drift_v4", "text_in_prompt"


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    missing = [row for row in rows if "first_token_s" not in row]
    if missing:
        raise SystemExit(f"{len(missing)} rows carry no first_token_s; rerun with the instrumented worker")
    return rows


def score(rows):
    memory = [row["first_token_s"][MEMORY] for row in rows]
    text = [row["first_token_s"][TEXT] for row in rows]
    outcomes = {}
    for row in rows:
        # A passage-level pair: the same question answered from memory and from text.
        outcomes.setdefault(row["passage"], []).append(
            (row["first_token_s"][TEXT] > row["first_token_s"][MEMORY], False))
    median_memory, median_text = statistics.median(memory), statistics.median(text)
    ratio = median_text / median_memory if median_memory else float("inf")
    p = sign_flip_p(list(passage_differences(outcomes).values()))
    report = {
        "kind": "PREREGISTERED EVALUATION (receiver prefill cost)",
        "questions": len(rows),
        "median_ttft_s": {MEMORY: round(median_memory, 6), TEXT: round(median_text, 6)},
        "mean_ttft_s": {MEMORY: round(statistics.fmean(memory), 6), TEXT: round(statistics.fmean(text), 6)},
        "text_over_memory_ratio": round(ratio, 3),
        "questions_where_memory_was_faster": sum(t > m for t, m in zip(text, memory)),
        "clustered_sign_flip_p": p,
        "bootstrap_95": clustered_bootstrap(outcomes),
        "sender_cost_s": {
            "median_translate_both": round(statistics.median([row["transport"]["translate_both_s"] for row in rows]), 6),
            "median_rdma_loop": round(statistics.median([row["transport"]["rdma_loop_s"] for row in rows]), 6),
        },
        "P1_receiver_prefill_is_cheaper": median_memory < median_text and p < 0.05,
        "P2_saving_is_material": ratio >= 2.0,
    }
    report["verdict"] = ("SUPPORTED" if report["P1_receiver_prefill_is_cheaper"] and report["P2_saving_is_material"]
                         else "PARTLY" if report["P1_receiver_prefill_is_cheaper"] else "NOT SUPPORTED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="rows.jsonl written by studio_mcdma_qa.py")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    report = score(load(arguments.rows))
    text = json.dumps(report, indent=1)
    print(text)
    if arguments.out:
        arguments.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
