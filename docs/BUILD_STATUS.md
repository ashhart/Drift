# Build status against the reviewed schematic

## Current reading, 21 September 2026

The tables below are the 20 September snapshot, not the current live readiness
verdict. Later exploratory MCDMA results are recorded in `FRONTIER.md`, sections
9-14. The reported six-case appointment demonstration was development-tuned;
it does not pass a fresh held-out evaluation or the prior-epoch M2 contract.
The takeover verified the dedicated Spark targets were still listening and
checked deployment files read-only, without rerunning model inference.

The local candidate now also verifies destination cache bytes before applied
receipts, preserves scheduler-confirmed final tap rows and implements an opt-in
fresh-child activation wake with numeric receipt binding. Its eight-module
connector manifest records the earlier five-module installation on both Sparks.
The integrated local suite passes 1,108 tests with four environment skips;
the plugin passes 143 tests and type checking. These are engineering checks,
not proof of improved recall or deployed KV-only OMP/Pi subagents. See
`GLM_CACHE_COMMIT.md` and `ACTIVATION_CHILD_WAKE.md` for the exact limits.
Local tests and candidate preparation do not deploy or qualify these changes.
Required real-adapter skips, runtime parity limits, scorer isolation and
approved compute budgets remain separate gates.

Historical phrases below such as "no text" describe the intended activation
payload, not proof that setup, control, files, timing or reports provide no
other channel. Neither the live scripts nor these tables establish a global
no-communication guarantee.

## Historical status

Updated 20 September 2026. "Mechanics" means built and tested on random tiny models or synthetic data on the
MacBook. Verdict labels follow AGENTS.md (PASSED / FAILED / BLOCKED / INVALID, plus NOT RUN).

## Live on real weights (19-20 September 2026)

Both frozen production models exchange memory purely as cache entries, in both directions, through their normal
serving stacks: GLM-5.3-Flash under vLLM on the two Sparks (TP=2, DFlash, fp8_ds_mla cache) and Qwen3.8-Flash-Next
under oMLX's runtime on the Studio (MLX 4-bit). No text, token ids or hidden states cross; neither model is trained.

| Path | How | Status |
|---|---|---|
| GLM tap | the owner's handoff export, parsed by `serving/glm53_handoff.py` (fp8_ds_mla -> canonical latents) | live, no server change |
| Qwen tap / inject | `serving/omlx_cache.py` through the cache objects (de-rotate / re-rotate, zero selector keys) | live |
| GLM inject | `serving/vllm_glm53_inject.py`, a subclass of the owner's connector that overwrites the latent pages of a placeholder span after the step that computed it; fail-closed on step boundaries | live (server restarted once with this class; revert command in /root/drift-live/restart_drift.sh) |
| Translators | `translate/stacked.py` + `scripts/live/fit_stacked_stream.py`: every reader layer reads all writer levels, ridge from streaming second moments, per-dimension variance restoration | fitted on ~282k paired tokens, both directions |
| GLM -> Qwen QA | `scripts/live/qa_eval.py`, prereg `configs/preregistration.live-qa-v2.json` | **PASSED**: exact match 0.875 (own-KV control 0.875, text 0.975, wrong memory 0.05, none 0.0), n=40 held-out vocabulary |
| Qwen -> GLM QA | `scripts/live/qa_eval_reverse.py`, prereg `configs/preregistration.live-qa-v2-reverse.json` | exploratory 4/4 on hand-written cases; preregistered run: see docs/agent-progress.md |
| Earlier tail-logprob test | prereg `configs/preregistration.live-glm-to-qwen.json` (first, per-level ridge translators) | PARTIAL (content-specific 19/24, not useful vs floor 15/24) |

Findings that changed the design: (1) a per-level linear map between the two models is capacity-limited, a map from
all writer levels is not; (2) the reader tolerates noise on injected K/V up to the signal's own std but not shrinkage
of the keys, and ridge shrinks, so variance must be restored; (3) a KV prefix alone, with the reader's recurrent
layers fresh, is enough for factual lookup; (4) on this vLLM build the first prefill step ends at
floor(L/64)*64-64, which fixes the placeholder layout and limits the reader's own prompt to <128 tokens for now.
What this is not yet: the gated append form of the spec (this is connector mode, a prefix), reasoning over transferred
memory, long contexts, the mailbox/thought channel, MCDMA transport, or a duel.

| Schematic block | Code | Status |
|---|---|---|
| Frozen model A / B, AttentionTap (canonical pre-RoPE K,V; MLA latent) | `adapters/qwen4_exp.py`, `adapters/glm5_next.py`, `adapters/toy.py` (kit dense) | Level-1 PASSED (tiny configs vs stock transformers 5.17). Real weights NOT RUN. |
| Identity replacement, private-state snapshot/restore | `adapters/base.py` | M0.2 PASSED (tiny). |
| ForeignKVBank / gated attention read / PositionAligner | kit `core/*` + adapters' gated block; `core/pool.py` (multi-writer) | PASSED (property + fuzz + adapter tests). |
| BridgeProjector → per-member translators, shared pool.v1, enrollment | `translate/pool.py` | Mechanics PASSED; fitted on random weights only. |
| Callosum: inproc / TCP streams, wire v1 + v2 | `transport/wire.py`, `transport/wire2.py`, `transport/backends.py`, `runtime/peer.py`, `scripts/peer_serve.py` | PASSED incl. two-process lockstep over loopback TCP and the per-host peer CLI. Cross-host NOT RUN. |
| Callosum: MCDMA | `transport/backends.py` guard, `transport/native_contract.h` | BLOCKED (needs Studio + Spark + pinned MCDMA APIs). |
| SyncController / one-epoch lag, N members | `runtime/worker.py`, `runtime/hive.py` | M2 in-process PASSED incl. checkpoints and poison rules. |
| Degeneration detectors | `eval/detectors.py` | PASSED. Thresholds not tuned (no data). |
| ThoughtWriter + marker + private mailbox branch | `mailbox/stream.py` (`MailWriter`), `train/mailbox_train.py` | M3.1 PASSED on all three adapter kinds; marker-salience training loop PASSED on toys. Real training NOT RUN. |
| Mailbox addressing, statuses, TTL, caps, metrics | `mailbox/stream.py` (`MailboxController`) | M3.2/M3.4 PASSED. |
| Counterfactual replay | `eval/replay.py` | M3.3 PASSED (mechanics). Held-out incorporation NOT RUN. |
| OFFLINE / drift-train | kit `train/*`, `train/behavior.py`, `train/probe.py`, `scripts/parity_real.py` | Regression, behavior and probe loops PASSED on toys; level-2 parity runner blocks without a preregistered tolerance. No real training. |
| READ-ONLY / audit + drift-eval | `eval/e1.py`, `eval/e2.py`, `eval/e3.py`, `eval/hive_experiments.py`, `eval/generate.py`, kit `eval/metrics.py` | Harnesses PASSED incl. E3 arms with budgets and the hive handoff / swap-in / departure arms. Formal E1–E4 NOT RUN. |
| Control plane: worker service (typed, HMAC) | `runtime/service.py`, `runtime/builders.py`, `docs/SERVICE_PROTOCOL.md` | PASSED on toy members. `from_manifest` refuses unqualified real members. |
| Control plane: omp-drift plugin | `plugin/omp-drift/` | PASSED: 26 tests, strict typecheck, live run against the real service. |
| E4 workspace / declared side channel | `eval/workspace.py` | PASSED (separate clones, metered reads, hidden tests outside clones). Duel NOT RUN. |
| MLX / Metal adapters (M4.1) | `adapters/mlx_qwen4_exp.py`, `adapters/mlx_glm5_next.py`, `adapters/mlx_fixtures.py` | Level-1 PASSED on tiny configs: cross-runtime stock parity torch↔MLX at 2e-6, foreign-path parity at 2e-5, 34 tests (1 xfail on a sparse-indexer tie). Requires `MLX_METAL_GPU_ARCH=applegpu_g16s` on M5 for fp32 (default kernels run TF32-class precision). Real weights NOT RUN. |
| M6 competition harness, registry | `compete/harness.py`, `registry.py` | Mechanics PASSED (signed submissions, invariants, budgets, unranked INVALID/BLOCKED). |
| M5 duel | — | NOT RUN (needs real models, the Duo text arm and hidden tests). |

## What is blocked on the owner

1. (resolved 19 Sep: owner granted both hosts; both models run in their serving stacks.)
2. A preregistered tolerance for quantized real-weight parity.
3. Calibration and evaluation corpora both models may read; hidden test sets under an audit root.
4. Budgets, `SERIES.md`, sandbox OS users (M−1.2).
5. MCDMA hardware time (M4.2/M4.3).
6. A precision policy for Metal: fp32 via the older kernel set, or a preregistered TF32/bf16 tolerance (docs/research/M4_MLX_NOTES.md).

## How to run

```bash
# pinned reference environment (transformers 4.56.2)
.venv/bin/python -m pytest -q
# next environment (transformers 5.17.0, mlx): adapters, translators and everything else
PYTHONPATH=. .venv-next/bin/python -m pytest -q tests/test_adapters_next.py tests/test_translate.py \
  tests/test_core.py tests/test_transport_sync.py tests/test_runtime_train_mail.py tests/test_properties.py \
  tests/test_wire2_pool.py tests/test_hive.py tests/test_mailbox.py tests/test_m1.py tests/test_service.py tests/test_workspace.py
```
