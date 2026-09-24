# Joint code task

Status: preregistered on 22 September 2026, before any scenario of this test was run.

## Question

Can GLM write working code for a shared project when the detail it needs was told only to Qwen, and that detail can reach GLM only as Qwen's memory over MCDMA?

## Why this test

Recall tests show one model naming a fact the other was told. This task asks GLM to use the fact to do work: the code only passes if the recalled detail is exact and used correctly.

## Design, fixed now

- 16 scenarios from seed 20260923. Each draws a fresh environment-variable name such as `SORREL_EGRET_KEY` from 400 combinations. Qwen alone is told that the project's API key lives in that variable, then writes about API-key practice.
- GLM is asked to write `get_api_key()`, which must return that variable's value and raise `KeyError` when it is unset. GLM first states the name it recalls, then gives the code.
- Three arms per scenario:
  - linked: the name can reach GLM only through Qwen's memory.
  - no_link: GLM must guess.
  - text: the name is in GLM's prompt, the ceiling.
- Qwen publishes its whole context at 3 copies into a 1,536-row reserve, the configuration of the reverse confirmation. No step uses the answer.
- Everything else is unchanged from the appointment runs: translators, deployment, loop settings and one greedy decode per arm.

## Measures

The primary measure is whether the code passes. The scorer runs GLM's last code block in an isolated Python process, with only the variable set, and checks both behaviours. Also reported: whether the recall line names the variable exactly.

## Hypotheses

- P1: linked passes more often than no_link. One-sided exact sign test over the scenarios where the two arms disagree, p < 0.05.
- P2: linked passes in at least 12 of 16 scenarios.

The outcome is SUPPORTED if P1 and P2 hold, PARTLY if only P1 holds, and NOT SUPPORTED otherwise. It is reported whatever it is. If the text arm passes in fewer than 12 of 16, the task itself is too hard for GLM as posed, and the linked result is reported with that caveat.

## Validity

The driver stops at the first invalid arm: a failed loop, a rejected loop report or a skipped publication. A stopped run is INVALID and repeats with a new seed once the cause is fixed.

## What this cannot show

One kind of detail, one function, agent-generated scenarios. It shows the channel can carry a project fact into working code. It does not show the models dividing a larger project between them.

## Command and cost

With the Sparks, the Studio and the MCDMA legs set in the environment:

```bash
PYTHONPATH=. .venv/bin/python scripts/live/demo_joint_code.py --seed 20260923 --scenarios 16 --out local/live/joint_code
```

48 arms at about 25 seconds each, roughly 20 minutes of GLM and Studio time.

## Result, 22 September 2026

Outcome: **NOT SUPPORTED**. Run from 21:40:55Z to 21:55:30Z at commit `e1eec4d`, with no invalid arm.

| Arm | Code passes | Recall names the variable exactly |
| --- | --- | --- |
| linked | 0 of 16 | 0 of 16 |
| no_link | 0 of 16 | 0 of 16 |
| text | 16 of 16 | 16 of 16 |

- P1 fails: no scenario separates linked from no_link.
- P2 fails: 0 of 16 against the required 12.

The text arm passed every scenario, so the task is within GLM's reach and the result is about the channel.

### Noticed after the fact, exploratory only

The first word of the variable reached GLM in 9 of 16 linked scenarios and 0 of 16 without the link. For example, `HOLLY_KINGFISHER_KEY` came back as `HOLLY_API_KEY` and `KESTREL_QUAIL_KEY` as `KESTREL_API_KEY`. GLM's prior supplied the rest. Part of the identifier crossed as memory, but the order and composition of its pieces did not, which is the same binding failure the reverse confirmation showed for streets. This count was not preregistered and is not part of the verdict.

Report SHA-256, private local copy: `46feccaadc57da9736bd7fe04dd287c07cd3fde16c03faaaae9f6c66f952b87e`.
