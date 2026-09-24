# Reverse recall confirmation

Status: preregistered on 22 September 2026, before any scenario of this test was run.

## Question

Does GLM name a fact that only Qwen was told, carried as Qwen's memory over MCDMA, more often than it names it by guessing without the link?

## Why a new test

The [selection ablation](APPOINTMENT_SELECTION_ABLATION.md) left this undecided. Its six scenarios put the true street only first or last in GLM's list. Unlinked GLM named the last street exactly when that was the answer, so the no-link control scored 3 of 6 and no config could separate recall from guessing.

## Design, fixed now

- 24 scenarios from seed 20260922. The true street sits in each of the four list positions in exactly 6 scenarios, in a shuffled order, so a positional guess scores at chance.
- Fresh street, shop and clinic names that no earlier run or prompt used, from `scripts/live/appointment_scenarios.py`.
- Qwen sends its whole context at 3 copies, the ablation's run C, into a reserve of 1,536 rows. No step of the transfer uses the answer.
- Every scenario runs linked and no_link, so each scenario is its own control.
- Translators, prompts and deployment are unchanged from the ablation: GLM with the Drift connector and `fp8_ds_mla`, Qwen under oMLX, one greedy decode per arm.

## Measures

The primary measure is `glm_named_only_true_street`: after the recall marker GLM names the true street and no other listed street. Also reported are `glm_correct` for the recommended shop, `qwen_correct` for the round trip, and the primary measure by list position.

## Hypotheses

- P1: linked beats no_link on the primary measure. One-sided exact sign test over the scenarios where the two arms disagree, p < 0.05.
- P2: the linked arm names only the true street in at least 18 of 24 scenarios.

The outcome is SUPPORTED if P1 and P2 hold, PARTLY if only P1 holds, and NOT SUPPORTED otherwise. It is reported whatever it is.

## Validity

An arm the loop admission rejects, or one with a skipped publication, is invalid. It is reported and never rescored. The driver stops at the first invalid arm, because a failed arm can leave remote processes that need checking, so a stopped run is INVALID and repeats with a new seed once the cause is fixed.

Amended before any run: the first version allowed up to 2 invalid arms, which the driver cannot produce because it stops at the first.

## What this cannot show

The names and seed are new, but the scenario template and prompts were tuned in development, and the agent generated the scenarios. This is a confirmation on fresh development-style data, not a held-out result. A held-out result needs a set the owner generates outside the agent's reach.

## Command and cost

With the head and peer Sparks, the Studio and the MCDMA legs set in the environment:

```bash
PYTHONPATH=. .venv/bin/python scripts/live/demo_appointment.py --scenarios 24 --seed 20260922 --balance --vocabulary fresh --selection all --prompt-copies 3 --reserve 1536 --conditions linked no_link --total-seconds 3600 --out local/live/reverse_confirm
```

48 arms at about 23 seconds each, roughly 20 minutes of GLM and Studio time.

## Result, 22 September 2026

Outcome: **PARTLY**. Run from 21:19:25Z to 21:38:30Z at commit `6cf987d`, with no invalid arm.

| Measure | Linked | No-link |
| --- | --- | --- |
| Named only the true street | 16 of 24 | 9 of 24 |
| Scenarios only this arm got right | 9 | 2 |
| Recommended the right shop | 16 of 24 | 9 of 24 |
| Qwen named GLM's recommendation | 6 of 24 | 0 of 24 |

- P1 holds: one-sided exact sign test over the 11 disagreeing scenarios, p = 0.0327.
- P2 fails: 16 of 24 against the required 18.

By list position the linked arm named the street in 3, 5, 4 and 4 of the 6 scenarios at each slot, so position no longer drives the result. Unlinked, GLM often guessed the same few names, which is why no-link reached 9.

So GLM recalls a fact that only Qwen was told more often than it guesses, on fresh names with balanced positions. It does so in two thirds of scenarios, below the three quarters fixed as the materiality bar.

### Noticed after the fact, exploratory only

Clinic names appear only in Qwen's fact and never in GLM's prompt. After the recall marker, GLM named the true clinic in 22 of 24 linked scenarios and 0 of 24 without the link. Most street misses were binding errors rather than missing content: GLM recalled the right clinic and paired it with an invented street, such as Ashgrove Lane or Elm Road, or turned the clinic name into a street, such as Larchmont Avenue. This count was not preregistered and is not part of the verdict.

Report SHA-256, private local copy: `8846d7e802de841e38db18bcd87ee63dd8db6e58e2810bb22bfb8213ccaf1b9a`.
