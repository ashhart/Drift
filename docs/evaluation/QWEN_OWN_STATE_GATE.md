# Qwen own-cache gate: attention rows plus recurrent state

Status: engineering gate, PASSED on 23 September 2026. Not a preregistered trial; it qualifies, for Qwen on the Studio, the mechanism the [GLM own-cache gate](GLM_OWN_STATE_GATE.md) qualified for GLM, and makes no claim about two models.

## Question

When Qwen gets its own cache of a passage, attention rows and recurrent state together, does it answer as it does with the passage as text?

## Setup

- Qwen3.8-Flash-Next under oMLX on the Studio, the same 24 validation questions as the GLM gate.
- One prompt per question frames a span with `@@DRIFT@@`; four arms answer greedily, up to 80 new tokens:
  - text: the passage's Qwen tokens stand in the span.
  - none: the span is empty.
  - rows: the head is prefilled, then Qwen's own K/V rows of the passage, from a separate read of head and passage, are appended at the passage's positions with the loop's `append_entries`, then the question follows.
  - rows_state: as rows, and each of the 36 linear-attention layers is also called on the passage's own inputs to it, captured in the same read, which advances its convolution window and recurrent state as reading the text would.
- Selector keys for the appended rows are zeros, exact while the context stays within the selector's budget.
- Command, on the Studio with oMLX's interpreter: `scripts/live/studio_qwen_state_gate.py --items out/gate_items.json --out out/qwen_gate.out.json`. Scored with `scripts/score_state_gate.py`.

## Result

| Arm | Answer present | Output identical to text |
| --- | --- | --- |
| text | 23 of 24 | |
| none | 0 of 24 | |
| rows | 24 of 24 | 12 |
| rows_state | 24 of 24 | 21 |

Rescored on 25 September with `drift/eval/answer_match.py`. The first scoring gave none 1 of 24, crediting the key `hare` inside the word "shared".

With both parts of its own cache Qwen reproduces the text arm's output in 21 of 24 questions, against 12 of 24 from rows alone. The advance uses Qwen's own layer code, so the path the cross-model state will take is qualified here.

## What it cannot show

Single-fact questions on Qwen's own cache. The per-layer embeddings Qwen mixes in from raw token ids keep their state from the head, and the gate does not measure what that costs.

Results SHA-256, private local copy: `dd59d568b3da460fc7db1780fb27e0daadf549b804a562066fda2871ff2614d2`. Cost: 95 seconds of Studio time, after one failed start: oMLX's patched layer passes speculative-verify options that the first capture wrapper did not accept.
