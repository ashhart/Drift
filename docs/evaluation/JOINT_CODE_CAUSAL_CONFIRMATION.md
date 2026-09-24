# Joint code task, causal confirmation

Status: preregistered on 23 September 2026, before any scenario of this test was run.

## Question

With GLM reading its own prompt after Qwen's memory has landed, can GLM write working code that needs a detail only Qwen was told, as well as it can when told the detail directly?

## Why a new test

The preregistered [joint code task](JOINT_CODE_TASK.md) passed 0 of 16 linked against 16 of 16 for text, NOT SUPPORTED. GLM then computed its prompt before any memory arrived. An exploratory causal run with seed 777 passed 14 of 16 (report SHA-256 `37a5b28ea7baf2fa07c2fa8dfd8a9f345f52cca2efce01f1f377ac0e67e7fa87`). That needs confirming on a new seed against the text arm.

## Design, fixed now

- 16 scenarios from seed 20260925. Each draws a fresh environment-variable name such as `SORREL_EGRET_KEY` from 400 combinations. Qwen alone is told that the project's API key lives in that variable, then writes about API-key practice.
- GLM is asked to write `get_api_key()`, which must return that variable's value and raise `KeyError` when it is unset. GLM first states the name it recalls, then gives the code.
- Three arms per scenario: linked with `--causal`, no_link, and text, where the detail is in GLM's prompt and the link is off.
- Qwen publishes its whole context at 3 copies into a 1,536-row reserve. No step uses the answer.
- Everything else is as deployed: translator `stacked3_rev`, the GLM bundle pinned at `5a11cc1` with the Drift scheduler, one greedy decode per arm.

## Measures

The primary measure is whether the code passes. The scorer runs GLM's last code block in an isolated Python process with only the variable set, and checks both behaviours. Also reported: whether the recall line names the variable exactly.

## Hypotheses

- P1: linked passes more often than no_link. One-sided exact sign test over the scenarios where the two arms disagree, p < 0.05.
- P2: linked passes in at least 12 of 16 scenarios, the bar of the original preregistration.
- P3, parity with text: linked passes in no more than one scenario fewer than text.

The outcome is SUPPORTED if P1, P2 and P3 hold, PARTLY if P1 holds without both others, and NOT SUPPORTED otherwise. It is reported whatever it is. If text passes in fewer than 12 of 16, the task is too hard as posed, and the result is reported with that caveat.

## Validity

The driver stops at the first invalid arm: a failed loop, a rejected loop report or a skipped publication. A stopped run is INVALID and repeats with a new seed once the cause is fixed.

## What this cannot show

One kind of detail, one function, agent-generated scenarios. It shows the channel can carry a project fact into working code. It does not show the models dividing a larger project between them.

## Command and cost

With the Sparks, the Studio and the MCDMA legs set in the environment:

```bash
PYTHONPATH=. .venv/bin/python scripts/live/demo_joint_code.py --seed 20260925 --scenarios 16 --causal --out local/live/joint_code_causal
```

48 arms at about 25 seconds each, roughly 20 minutes of GLM and Studio time.

## Result, 23 September 2026

Outcome: **SUPPORTED**. Run from 23:48:25Z on 22 September to 00:02:36Z at commit `303d6e5`, with no invalid arm. Scored by `scripts/score_causal_confirmation.py`, committed before the run finished.

| Arm | Code passes | Recall names the variable exactly |
| --- | --- | --- |
| linked | 15 of 16 | 15 of 16 |
| no_link | 0 of 16 | 0 of 16 |
| text | 16 of 16 | 16 of 16 |

- P1 holds: linked alone passed in 15 scenarios, no-link alone in none, one-sided exact sign test p = 3.1e-05.
- P2 holds: 15 of 16 against the required 12.
- P3 holds: linked is one scenario below text.

The one miss hedged: GLM wrote code for a variable name it was unsure of, which was wrong, and asked the user to confirm the name.

The preregistered [joint code task](JOINT_CODE_TASK.md) scored 0 of 16 linked on the same design. The only change is that GLM reads its prompt after Qwen's memory lands. GLM now writes working code that depends on a detail only Qwen was told, nearly as often as when it is told the detail directly.

Report SHA-256, private local copy: `2a59d80b27c147a97ec3b774f71711f0e9772fbf023266f15291ebbe027214d2`. Cost: 14 min 11 s of GLM and Studio time for 48 arms.
