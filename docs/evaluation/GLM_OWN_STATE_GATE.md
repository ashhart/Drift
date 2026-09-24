# GLM own-cache gate: attention rows plus recurrent state

Status: engineering gate, PASSED on 22 September 2026. Not a preregistered trial; it qualifies the mechanism in [GLM recurrent state](../reference/glm/GLM_RECURRENT_STATE.md) and makes no claim about two models.

## Question

When GLM gets its own cache of a passage, attention rows and recurrent state together, does it answer as it does with the passage as text?

## Setup

- Server: GLM-5.3-Flash on the two Sparks, restarted with `--scheduler-cls glm_prefill_scheduler.DriftScheduler` and the connector bundle pinned at `5a11cc1`.
- Items: 24 questions, two from each of the first 12 validation passages in `local/live/train_taps/triples.json`. None come from an evaluation set.
- One prompt per question frames a span with `@@DRIFT@@`. Four arms answer greedily, up to 80 new tokens:
  - text: the passage's GLM tokens stand in the span.
  - none: the span is empty.
  - rows: a reserve of exactly the passage's length takes GLM's own exported latents as the first publication, and the prefill stops at the reserve's end, so the question is computed after the memory lands.
  - rows_state: as rows, plus each rank's own exported recurrent state written at that stop.
- Command, on the head Spark: `spark_run.sh glm_state_gate.py --items gate_items.json --out gate_full.out.json --worker <rank-1 host>`. Scored with `scripts/score_state_gate.py`.

## Result

| Arm | Answer present | Output identical to text | Median characters shared with text before diverging |
| --- | --- | --- | --- |
| text | 24 of 24 | | |
| none | 0 of 24 | | |
| rows | 23 of 24 | 1 | 8 |
| rows_state | 24 of 24 | 13 | 38 |

The one rows miss is the old failure: GLM said it had no memory of the report. With its state it answered the code. Where rows_state differs from text, it diverges after tens of characters, and the answer is the same. That fits numerical drift from re-quantized fp8 rows and different prefill chunk boundaries, not a different reading.

## What it shows

- The mechanism works: both parts of GLM's own cache reproduce text-level answers, and half the outputs word for word.
- Ordering matters as much as the state. With the question computed after the memory, own rows alone reach 23 of 24 single-fact answers. The live loop has never run that way, because it sends GLM's request while Qwen is still reading.

## What it cannot show

These are single-fact questions on GLM's own cache. It says nothing about Qwen's translated rows, about a translated state, or about questions that bind several facts.

Results SHA-256, private local copy: `219e14600c5201ea5df222af41c9397addbca420358fa64e5623d15690eeb761`. Cost: 4 min 45 s of GLM time, after two restarts of about 6 minutes each and one failed attempt described in the progress log.
