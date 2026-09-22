# Why the recorded answers are below 100 percent

Status: diagnosis in progress, not an accuracy repair or a new evaluation.
Base: `3a57ba74effdf5029bffa6144294d826d364bf4c`, with local uncommitted changes.
Backbones, translator artifacts, questions and historical scoring are unchanged.
This investigation replays existing records and translation arithmetic; it has
not launched new model inference, a benchmark campaign or a remote deployment.

## Forward, GLM to Qwen

The existing MCDMA report contains 234 correct translated-memory answers out of
240, against 236 with full text and 224 with Qwen's own KV-only memory.
The six misses are not one uniform transport failure:

| Miss category | Full text | Own KV | What the existing evidence isolates |
| --- | --- | --- | --- |
| count | correct | correct | Additional translated-memory failure in the MCDMA run |
| person | correct | correct | Additional translated-memory failure in both SSH and MCDMA runs |
| animal | wrong | wrong | Shared failure even when cross-model translation is absent |
| colour | wrong | wrong | Shared failure even when cross-model translation is absent |
| ingredient | wrong | correct | Mixed reader/representation outcomes |
| room | correct | wrong | Mixed reader/representation outcomes |

All six reference values occur once as a complete word or number in the source
passage. Simple Unicode, whitespace and alphanumeric normalization recovers none
of the six. In two failures, the response contains another annotated fact from
the same passage: a weight rather than a count, and the second person rather than
the responsible person. These are evidence of attribute confusion in the output,
not proof that the fact's activation never arrived. Source presence alone does
not establish that the generated prose unambiguously preserves the fact sheet's
intended role; the generator checks substring presence, not semantic attribution.

One apparent v3 recovery is a scoring false positive: the expected room number
occurs only inside a longer integer. Historical scores were not edited, and this
finding is not used to invent extra v4 successes. Neither raw answers nor
passages are copied into this document.

The SSH and MCDMA runs are not identical except for transport. They also differ
in float16 versus float32 source staging, NumPy versus MLX translation, and
absolute rotary positions. Their relative prefix/query distance is the same,
but absolute rotary arithmetic differs. Wrong-memory controls also select
different neighboring passages. Five v4 verdicts change between runs: two
recoveries and three new misses. The captures do not establish identical native
inputs, so those changes alone do not prove decoder nondeterminism.

### Arithmetic replay, without model inference

The six corresponding original SSH source captures remain available; the MCDMA
activation bytes were not retained. A private replay selected those captures
without exposing their contents and checked all three frozen translator hashes.
Across 914 source tokens, 933 emitted rows and 11,464,704 KV elements:

| Comparison | Relative L2 difference | Maximum absolute difference |
| --- | --- | --- |
| Saved SSH output versus current NumPy reference | 0 | 0 |
| NumPy reference versus MLX GPU | 0.0007130133 | 0.03125 |
| NumPy reference versus MLX CPU | 0.0000033664 | 0.001953125 |
| Full-passage versus eight-row MLX GPU chunks | 0.0000571857 | 0.0078125 |

Fanout order and row counts match in every comparison. This is measured numeric
drift, not demonstrated causation of a wrong answer. The replay used corresponding
SSH captures, not unavailable MCDMA bytes, and no inference scores were produced.

## Reverse, Qwen to GLM

Existing runs 5, 6 and 7 score 21, 18 and 20 out of 30. Across them, 18 items are
always correct, nine always wrong and three vary. Both ranks' applied file hashes
are identical for all 30 items between runs 6 and 7. They differ between runs 5
and 6, so the earlier claim of identical applied files across all three runs is
not established; repacking may explain the difference, but tensor identity was
not checked. Four of the nine stable misses have identical full answer text.

Two verified implementation mismatches are candidates for the stable misses:

1. Reverse fitting selects shared tokenizer endpoints, but reverse inference
   translates every Qwen token row, including intermediate digit/name subtokens
   without a direct target row in that fitting procedure. Previous attempts to
   drop or merge them worsened results, so deleting rows is not a justified fix.
2. The connector overwrites MLA latents but leaves selector keys and recurrent
   state derived from placeholders. Of 30 recorded prompts, 27 exceed the
   checkpoint's recorded 2,048-entry selector budget, including eight of the
   nine stable misses. That can make foreign-row selection consequential, but
   actual selected-row coverage was not captured, and the ninth miss is below
   the budget. This is not yet a causal explanation for all nine.

Every item also leaves some reserved placeholder rows unwritten. Their count
does not cleanly separate correct and incorrect items. Successful file and rank
receipts prove delivery/application, not selection or semantic use.

## Diagnostic repair

The reverse worker previously discarded completion reasons and exact token
usage, mishandled usage-only SSE footers, and accepted EOF without `[DONE]`.
Consequently old reports cannot establish whether a 64-token cap truncated a
particular wrong answer. A focused parser now requests and preserves terminal
reason and usage, refuses incomplete streams, and reports the first nonempty
text arrival rather than an empty metadata frame. Both launchers stage the parser
and the continuous summary preserves its metadata. This local change removes
diagnostic blindness; it does not retroactively identify truncation or fix recall.

The previously corrected live KV-position policy and stale-logit characterization
concern the continuous/subagent paths. They do not explain all six staged-reader
misses. A later local repair captures and finalizes the accepted partial tap
tail and adds receipt-bound fresh-child wake, but neither is deployed or
checkpoint-qualified. Generic resumed-subagent wake remains open; details are
in `docs/reference/omp/ACTIVATION_CHILD_WAKE.md` and `docs/reference/glm/GLM_CONNECTOR_DEPLOYMENT.md`.

## Current boundary

PASSED: deterministic aggregation and byte-level reference translation replay.
FAILED: 100-percent answers in the existing forward and reverse records.
BLOCKED for causal attribution: missing MCDMA activation captures, native
selector-coverage evidence and old completion reasons. A focused native replay
at those boundaries is needed to distinguish the remaining mechanisms; static
inspection cannot select one honestly. Do not retrain on these evaluation misses,
silently change historical scoring, or assert that a small code patch guarantees
100-percent answers in either direction.
