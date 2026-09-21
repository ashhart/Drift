# Prefix-cache reads for linked GLM sessions (proposal, prepared offline)

Status: integrated locally; native qualification still requires a verified deployment.
The canonical source closure is `configs/glm_connector_deployment.json`; the older
branch candidate and private deployment observations have been archived outside the repository.
Historical test counts below describe the original patch, not the current checkout.

## Why linked turns recompute everything today

| Layer | Where | Effect |
|---|---|---|
| Random cache salt per request | `glm_factory.py`, `glm_restore_turn.py`, live scripts | no block hash ever repeats |
| No-store flag on every request | same places | nothing is stored, so nothing could be hit |
| `skip_reading_prefix_cache = True` | connector `on_new_request` | hits refused even if blocks existed |
| Reserved span at the very start of the system text | `glm_restore_prompt.reserve_prompt` | no stable tokens precede the span anyway |

Each layer exists for a reason that still holds: the span's blocks are overwritten with foreign memory and must belong to
the request alone, and nothing holding foreign memory or a private transcript may be stored for another request to hit.

## The change

Connector (`drift/serving/vllm_glm53_inject.py`), live sessions only, default unchanged:
- `kv_transfer_params.drift_prefix_reuse: true` (the literal) leaves cache reads enabled. At the request's FIRST scheduling
  vLLM's hit length is compared with `drift_reserve_start`; a hit that reaches the span fails closed with
  `LiveReceiverError('PREFIX_HIT', hit=…, start=…)`: no write, no tap, session poisoned, numeric diagnostics only.
- Any other value refuses the request's live use and keeps reads disabled. One-shot `drift_inject` requests are unchanged.
- Separate fix found on the way: a live request that vLLM preempts and reschedules has lost whatever was written into its
  span and would silently continue on placeholder latents. It is now refused with `PREEMPTED`.

Client policy (`drift/serving/glm_prefix_reuse.py`, pure functions, not yet wired into `TurnRestorer`):
- `reserve_after_system`: the marker run follows the own system text; the existing `verify_prefix` proves the insertion.
- `session_salt`: one salt per worker session, so the stable prefix hashes identically on every turn and only there.
- `linked_extra`: linked turns may READ but still never WRITE the cache (`skip_writing_prefix_cache` stays on).
- `warm_request`: a no-link request (no `kv_transfer_params`, one output token) is the only thing that stores the prefix.
- `reusable_tokens`: upper bound = common token prefix of warm and linked prompts, cut at the span, rounded down to the
  engine's 3,584-token unit. A stable prefix shorter than one unit gains nothing.

## What is NOT established

- That hits materialise on this build. The owner's hybrid patch serves hits at 3,584-token alignment where recurrent state
  was checkpointed, and drops a block for the drafter; whether a one-token warm request leaves a usable checkpoint at the
  boundary is an engine fact. Measure `usage.prompt_tokens_details.cached_tokens` on the linked turn.
- Any saving. Counted input tokens do not change (the API reports submitted tokens); the expected effect is prefill TIME and
  compute, to be measured as time-to-first-token with and without reuse on identical turns. No ratio is claimed.
- Privacy of the stored warm prefix: it contains the own system text and tool schema under a per-session salt and stays in
  the GPU cache until evicted. It contains no foreign memory and no conversation. `/reset_prefix_cache` is disabled on this
  server (`GLM53_EXPOSE_CACHE_RESET=0`).
- Tensor-parallel agreement is unchanged by this patch: both ranks see the same scheduler decision.

## Proposed qualification, after review and re-pin (bounded, no training, no restart beyond the bundle cutover)

1. Re-pin, stage, staged-import receipts, coordinated cutover (see `GLM_CONNECTOR_DEPLOYMENT.md`).
2. No-link: warm request, then the same prompt twice with a stable salt; record `cached_tokens` and time-to-first-token.
3. Linked, span after the system text, `drift_prefix_reuse: true`: confirm `cached_tokens <= reserve_start`, all rank
   receipts, and identical answers to a run with reuse off (greedy; this stack is not bit-deterministic, so compare the
   scoring outcome, not bytes).
4. Negative control: span placed inside the first unit with a warmed prefix must produce `PREFIX_HIT` and no write.
5. Preemption: not forced on the production server; covered by the unit test only.

Tests: `tests/test_glm_prefix_reuse.py` (6) — default still forbids reads; only the literal `true` opts in; a hit that stops
before the span is accepted and memory lands in the request's own block while the cached block stays untouched; a hit
reaching the span fails closed with numeric diagnostics; preemption is refused and latches; client policy and the
reusable-token bound. Full suite on this branch: 827 passed, 3 skipped, 2 failed (the manifest tripwire described above).

## Review response (2026-09-21)

A review of `a51a31f`/`da48055` blocked it on the five findings summarized below; the original review is archived privately.
All five are corrected on this branch; its six failing regressions now pass unchanged against this worktree, and
`tests/test_glm_prefix_reuse.py` carries equivalent cases.

| Finding | Correction |
| --- | --- |
| P1 resumed request arriving in `scheduled_new_reqs` bypassed the preemption refusal | An already-seen live request arriving as new is treated as a resume: blocks replaced, turn refused with `PREEMPTED`. Both resume representations are covered. |
| P1 reuse admission did not enforce no-store | The connector admits `drift_prefix_reuse` only when the request's resolved `skip_writing_prefix_cache` is the literal `True`; otherwise it refuses with `NO_STORE` and disables cache reads. |
| P2 helpers accepted unscoped salts and malformed spans | `warm_request` and `linked_extra` require a salt produced by `session_salt`; `linked_extra` requires a named session and a typed, bounded span; `warm_request` rejects a system message that already carries the reservation. |
| P2 reusable-token bound could exceed the prefix | `reusable_tokens` requires a positive integer alignment, a non-negative span start and integer token ids. |
| Style | Module and function docstrings and connector comments are one line; the rationale lives in this document. |

The no-store attribute was read from the deployed image, read-only, with no inference: the owner's build resolves
`SamplingParams.skip_writing_prefix_cache` or `vllm_xargs["skip_writing_prefix_cache"]` into `Request.skip_writing_prefix_cache`,
and the block pool skips stores when it is true. The owner's kill switch `GLM53_APC_NO_STORE=0` resolves it to false, in which
case this connector now refuses reuse instead of reading a cache that linked turns could also write.

Why reads stay safe, previously in code comments: cached blocks may serve only tokens BEFORE the reserved span, because the
span's blocks are overwritten with foreign memory and must belong to this request alone. The guard runs at first scheduling,
where `num_computed_tokens` is the engine's own cache hit. A preempted request loses whatever was written into its span, so it
is refused rather than continued on placeholder latents. Linked requests never store, so no block that held foreign memory, a
private transcript or the span can be served to a later request; only the separate no-link warm request stores the stable prefix.

Still open, unchanged by this response: whether the deployed scheduler uses the resumed-as-new shape is UNVERIFIED; the salt's
randomness is the caller's responsibility; the integration conditions in the review (shared placement option with
`glm_restore_prefix_child.py`, warm-once state and budgets in `BankRestorer`) are not implemented here; nothing is deployed
and no host measurement exists.

Re-pinned candidate after the corrections: source `e3d112c`, manifest SHA-256
`20da78f8703c28eec9ff00f7a2524b7548fee1663f95c1351731794dba9fe49c`, locally prepared bundle
`26bde7922ef3438feabb683ffbc7154d72e5c1287c05d37a26646e784728b21e`. The read-only host check reports BLOCKED, as it must:
both ranks run the deployed bundle, not this candidate. Reference suite: 832 passed, 3 environment skips, and the same two
deployment-manifest tripwire failures as before, which exist because the default manifest rejects the changed connector.
