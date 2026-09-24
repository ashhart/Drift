# Appointment selection ablation

Status: exploratory. Planned before any run, on development scenarios only.

## Question

The appointment demonstration sends only the rows of the appointment sentence. The controller finds them by matching the fact text, tiles them 12 times and sends nothing else. What crosses is KV, not text, but the choice of rows used the answer.

Does GLM still recall the appointment when Qwen sends its whole context, so that no step uses the answer? And how many copies does that need?

## Fixed for every run

- The six development scenarios from seed 7, the same ones as `demo_run2`.
- A reserve of 1,536 rows, which fits the largest config: `all` is 111 rows per scenario, so 12 copies is 1,332.
- The deployed translators and prompts, with GLM running the Drift connector and `fp8_ds_mla`.
- One greedy decode per arm.

## Configs

| Run | Selection | Copies | Rows sent | Conditions |
| --- | --- | --- | --- | --- |
| A | oracle | 12 | 264 | linked, no_link |
| B | all | 12 | 1,332 | linked |
| C | all | 3 | 333 | linked |
| D | all | 1 | 111 | linked |

Run A repeats `demo_run2` on today's deployment. The no-link arm runs once, because it publishes nothing and so selection and copies cannot change it.

## Measures

The primary measure is `glm_named_only_true_street` in the linked arm, which requires the true street and no other listed street. Also reported are `glm_recalled_true_street` for comparison with earlier runs, `glm_correct` for the recommended shop, and `qwen_correct` for the round trip.

GLM sees four streets, so chance for a street is about 1 in 4.

## Reading the outcome, fixed in advance

- If B scores 5 or 6 of 6 on the primary measure while no-link scores 1 or fewer, sending the whole context works at 12 copies and the oracle was not doing the work.
- If B scores 2 or fewer of 6, the oracle selection was doing the work, and a general system needs a way to choose what to send.
- A score of 3 or 4 of 6 is undecided at this sample size. The next step would be more scenarios, not a conclusion.

C and D answer the copies question under the same rule. A against C is the closest match in total rows, 264 against 333, with the fact concentrated in A and diluted in C.

## What this cannot show

Six development scenarios, one decode each, all used during earlier tuning. The run can rule the approach in or out for this setup. It cannot measure reliability, and none of it is a held-out result.

## Results, 22 September 2026

All 30 arms were valid, with no skipped publication and no rejected loop. Each run took 2 to 5 minutes, about 23 seconds per arm. Run A used commit `14c5f34` and runs B to D used `9fc878f`, which differs only in tests.

| Run | Selection | Copies | Rows sent | Primary measure | Linked only | No-link only | One-sided p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no-link | none | none | 0 | 3 of 6 | | | |
| A | oracle | 12 | 264 | 6 of 6 | 3 | 0 | 0.125 |
| B | all | 12 | 1,332 | 5 of 6 | 3 | 1 | 0.3125 |
| C | all | 3 | 333 | 6 of 6 | 3 | 0 | 0.125 |
| D | all | 1 | 111 | 5 of 6 | 3 | 1 | 0.3125 |

The p values come from a one-sided exact sign test over the scenarios where the linked and no-link arms disagree.

### Against the rules fixed in advance

A positive reading needs no-link at 1 or fewer. No-link scored 3 of 6, so no run qualifies, and no run of B, C or D fell to 2 or fewer. The outcome is undecided at this sample size, and the plan's own next step applies: more scenarios, not a conclusion.

### Why no-link scored 3

The generator draws the true street's place in GLM's list at random. With seed 7 it came out first in scenarios 0, 1 and 3 and last in 2, 4 and 5. Without the link GLM never named the first street, and it named the last one in exactly the three scenarios where that was the answer. This is a guessing pattern meeting a small sample, not a leak. The no-link arm publishes nothing, and Qwen's follow-up answered unknown in all six.

### Noted after the fact, exploratory only

In the three scenarios where the unlinked guess failed, every config recalled the true street, including a single copy of Qwen's whole context. Both misses came on scenarios where the unlinked guess was right anyway. In B scenario 2 GLM described an appointment on Elm Road, a street that is not in its list. In D scenario 4 it said it recalled nothing.

Qwen named GLM's recommendation in 2 of 6 for A and 1 of 6 for B, C and D. The forward half of this loop mostly fails, and none of these runs was designed to measure it.

### Evidence

Report SHA-256, private local copies:

- A `a5716240c5493a03ea01c6496a424ba34008645e083086ad1d6d77c45d6b8b1c`
- B `6518b16fa6bfe197a5968a07761c61d45c6d9c527a2b8262493c4c7efded3369`
- C `5e6d188caaa6e3e9990c749fba038293649f23ecda292d14dfa91ee015c5e112`
- D `9802e84ffec345f32b4abff387a3b7ba177fc4ae809e2b19e43195d782c7ec3a`

Two earlier attempts at run A failed before scoring and are kept. The first stopped at startup because the Studio scripts held placeholder link addresses, fixed in `7a04621`. The second crashed on the first decode step because the rephase called a cache method the deployed oMLX lacks, fixed in `f2ed3eb`.

### Next

A follow-up on fresh scenarios with the true street's position balanced across the four slots, whole context at 3 copies against no-link, planned in its own document before it runs.
