# Causal recall confirmation

Status: preregistered on 23 September 2026, before any scenario of this test was run.

## Question

When GLM reads its own prompt after Qwen's memory has landed, does it recall a fact only Qwen was told as well as it does when the user tells it the fact directly, and better than it guesses?

## Why a new test

The [reverse recall confirmation](REVERSE_RECALL_CONFIRMATION.md) scored 16 of 24, PARTLY. GLM then computed its whole prompt, question included, before any memory arrived. The [causal development runs](CAUSAL_LOOP_DEVELOPMENT.md) changed only that ordering and scored 12 of 12 on development scenarios. That needs confirming on inputs no run has seen, against the text arm.

## Design, fixed now

- 24 scenarios from seed 20260924, balanced so the true street sits in each of the four list positions 6 times.
- The `confirm2` vocabulary in `scripts/live/appointment_scenarios.py`. No run used these names before this test.
- Qwen sends its whole context at 3 copies into a 1,536-row reserve, selection `all`. No step of the transfer uses the answer.
- `--causal`: GLM's session waits for Qwen's first publication, and the Drift scheduler stops GLM's prefill at the reserve's end.
- Three arms per scenario:
  - linked: the fact can reach GLM only through Qwen's memory.
  - no_link: GLM must guess.
  - text: the user's message to GLM starts with the fact, and the link is off. This is the bar.
- Everything else is as deployed: translator `stacked3_rev`, the GLM bundle pinned at `5a11cc1` with the Drift scheduler, Qwen under oMLX, one greedy decode per arm.

## Measures

The primary measure is `glm_named_only_true_street`: after the recall marker, GLM names the true street and no other listed street. Also reported: `glm_correct` for the recommended shop, `qwen_correct` for the round trip, and the primary measure by list position.

## Hypotheses

- P1: linked beats no_link on the primary measure. One-sided exact sign test over the scenarios where the two arms disagree, p < 0.05.
- P2: linked names only the true street in at least 21 of 24 scenarios.
- P3, parity with text: linked scores no more than one scenario below text on the primary measure.

The outcome is SUPPORTED if P1, P2 and P3 hold, PARTLY if P1 holds without both others, and NOT SUPPORTED otherwise. It is reported whatever it is. If text scores below 21 of 24, the task is too hard as posed, and the result is reported with that caveat.

## Validity

An arm the loop admission rejects, or a linked arm with a skipped publication, is invalid. The driver stops at the first invalid arm, so a stopped run is INVALID and repeats with a new seed once the cause is fixed.

## What this cannot show

The names and seed are new, but the agent wrote them, and the scenario template and prompts were tuned in development. This is a confirmation on fresh development-style data. A held-out result needs the owner's names and seed, per the [held-out protocol](HELD_OUT_PROTOCOL.md).

## Command and cost

With the Sparks, the Studio and the MCDMA legs set in the environment:

```bash
PYTHONPATH=. .venv/bin/python scripts/live/demo_appointment.py --scenarios 24 --seed 20260924 --balance --vocabulary confirm2 --selection all --prompt-copies 3 --reserve 1536 --conditions linked no_link text --causal --total-seconds 4200 --out local/live/causal_confirm
```

72 arms at about 25 seconds each, roughly 30 minutes of GLM and Studio time.

## Result, 22 September 2026

Outcome: **SUPPORTED**. Run from 23:21:38Z to 23:48:25Z at commit `597dd41`, with no invalid arm. Scored by `scripts/score_causal_confirmation.py`, committed before the run finished.

| Measure | Linked | No-link | Text |
| --- | --- | --- | --- |
| Named only the true street | 24 of 24 | 6 of 24 | 24 of 24 |
| Recommended the right shop | 22 of 24 | 6 of 24 | 24 of 24 |
| Qwen named GLM's recommendation | 2 of 24 | 0 of 24 | 0 of 24 |

- P1 holds: linked alone right in 18 scenarios, no-link alone in none, one-sided exact sign test p = 4e-06.
- P2 holds: 24 of 24 against the required 21.
- P3 holds: linked equals text, 24 and 24.

Linked named the street in all 6 scenarios at each list position. The two shop misses both named the true street, then reached the 260-token limit before writing the recommendation line. Neither named a wrong shop.

GLM now recalls a fact only Qwen was told, carried as Qwen's memory over MCDMA, as well as when the user tells it the fact directly. Against the earlier confirmation's 16 of 24, the only change is that GLM reads its question after the memory lands.

Qwen's follow-up is unchanged at 2 of 24. That is the forward direction, whose recurrent layers still never see GLM's memory.

Report SHA-256, private local copy: `5fa7096499646576f1102e96c8b3f1a9efeaac07f48cc8ff9e5395d9f2cfdfa6`. Cost: 26 min 47 s of GLM and Studio time for 72 arms.
