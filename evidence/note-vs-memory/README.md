# Note versus memory: the evidence

This directory holds everything behind one claim: that a question-blind written handover
drops facts the receiving model could have used, and that a translated memory link
preserves them.

The criteria were fixed in `preregistration.json` **before any note was written or
scored**. The memory arm was measured before that file existed, and only the note arms
were new; that is stated in the preregistration itself rather than implied.

## The result

| Arm | Exact match over 240 questions |
| --- | --- |
| GLM's translated memory, over MCDMA | 0.975 |
| The full passage as text (control) | 0.9833 |
| GLM's note, at most 50 words (median 44) | 0.8083 |
| GLM's note, at most 25 words (median 25) | 0.65 |

Both preregistered hypotheses were met: memory minus the 50-word note is 0.167 against a
required 0.15, and memory minus the 25-word note is 0.325 against a required 0.30, with
clustered sign-flip p-values of 7e-10 and 5e-18. Verdict: `SUPPORTED`.

## Re-score it yourself

```bash
python scripts/score_note_vs_memory.py --compare
```

That recomputes every number from the recorded model outputs and exits nonzero if the
published `report.json` does not reproduce. `tests/test_note_vs_memory_evidence.py` runs
the same check in CI, so the claim is verified on every commit.

## What is here

| File | Contents |
| --- | --- |
| `preregistration.json` | The design, hypotheses, thresholds and stated limits, fixed in advance |
| `passages.json` | The 120 synthetic passages, each carrying six to eight facts |
| `notes.jsonl` | Every handover note GLM wrote, at both budgets, with its word count |
| `answers.jsonl` | Every answer Qwen produced from a note |
| `memory_arm.report.json` | The 240 questions with the output of all five conditions and both controls |
| `report.json` | The scored summary this directory reproduces |
| `SHA256SUMS` | Digests of the files above |

## What this does not establish

The passages are short and synthetic. A longer note or the full passage as text recovers
the facts, and the text control scores 0.9833, above the memory arm. The claim tested is
about what a bounded, question-blind handover drops, not about text in general.

The note writer overran its own budget on some notes, including one 25-word note that ran
to 123 words. That is left in the data rather than trimmed, and it works against the
result rather than for it.

Re-scoring checks the arithmetic and the decision rule against recorded outputs. It does
not re-run the models, and it does not establish that the effect holds on other corpora,
longer contexts, other model pairs or real work.
