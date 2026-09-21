# Bounded native GLM generated-prefix outbox

The optional collector is a local engineering connection to the existing connector's
canonical tap files; it does not establish complete final-cache export, causal use,
source attribution or continuous Duo collaboration.
Default native restoration still sends `drift_tap: false`.
No deployed connector, serving flags, native chat parser or worker own-input frame changes.

## Existing verified hook and lifetime

Read-only inspection found deployed/local matching sources:
`vllm_glm53_inject.py` SHA-256
`d5f525973e29db9a3ccc7a33b1402322e98330e78d0a40302d48bf67cafc1967`
and `live_receiver_glm.py` SHA-256
`b1ce61bee7c9652212e355d1e14159976ac599761049006a5403a47a6e32fa4b`.
The deployed base `glm53_handoff_connector.py` hash is
`b0f0d56a373e58fb97f680d7210d29e5a9a55c1b3f050181415b7f56fd669a70`.
The existing per-request `kv_transfer_params.drift_tap` enables rank-zero canonical
NoPE MLA exports under `tp-live-out/<fresh-session>/<sequence>.npz`.
Each atomic file contains finite float16 `lN` arrays and numeric `start`/`stop` cursors.
No token IDs or generated text travel in those files.

The receiver exports only up to `LiveStep.before`, the already verified scheduler
cursor, and only when the configured tap threshold is reached (source default eight).
It does not flush the final short tail.
The scheduler-side `finished` marker contains only `failed` and `writes_scheduled`,
not a final exported cursor or worker-side drain acknowledgement.
Both base finish hooks return `False, None`; deployed vLLM immediately frees request
blocks unless a connector explicitly delays freeing them.
The deployed scheduler source hash was
`d68555574a5e23f872f401dc72135266df9ba9618722aaf867e2b2e74f1c38e5`.
Reading old block IDs after HTTP completion is therefore unsafe.
The base `glm53_handoff` option is not a substitute: it forces `max_tokens=1` and its
full blobs include prompt IDs and other private fields inappropriate for this channel.

## Explicit opt-in

A linked restoration worker may add an `outbox` object to its owner configuration:
`memory_root`, `source_worker`, `target_worker`, `recipe_sha256`, `max_raw_rows`,
`max_raw_bytes`, `max_rows`, `max_bytes`, and `max_publications`.
The root must already be a canonical private mode-0700 controller directory.
Source/target labels must reverse the incoming recipe's ownership labels.
The recipe hash is declared future translation identity; this collector loads no
translator and reports `translation: NOT_PERFORMED`.
The existing worker's `max_turns` also bounds the collector's fresh session claims.
Owner manifests must pin this configuration before any authorized dispatch.

`TurnOutbox.begin` binds the fresh restoration session, verified prompt length,
reserved-span endpoint and proposed generation limit before publication staging.
Native input counting may reduce that limit; the final collector uses the actual
admitted `max_tokens`, not the original larger proposal.
The optional helper enables the already deployed tap flag without changing the
native `/v1/chat/completions` request structure or tool event parser.
`BankRestorer` carries the same outbox helper through versioned snapshot binding.

At turn completion, the collector waits for the existing finished marker within the
same worker deadline, refuses rank errors, temporary files, sequence gaps and range
mismatches, snapshots bounded source files privately and verifies their exact hashes.
It writes selected canonical arrays to immutable private files and atomically publishes
an aggregate manifest; completed-turn restoration receipts contain its path/hash and
aggregate counts, never activation contents or model text.
Each manifest also records the original publication sequence, byte count and hash,
so source-side export overhead is visible even when no generated rows are selected.
Raw row/byte/publication limits and selected row/byte limits are separate.
The raw limit includes the reconstructed prefill; it can be much larger than the
selected generated prefix and must be explicitly budgeted.
Limits bound collector admission/consumption; they do not install a native-writer
filesystem quota or retroactively prevent an oversized connector export.

## Generated-only scope and incomplete final tail

Selection retains only source positions at or after the verified `prompt_tokens`
boundary, excluding both restored foreign placeholders and repeated reconstructed
prefill on every tool turn.
This also excludes newly introduced own tool results, notes and user instructions,
since those are part of that turn's prompt.
The manifest therefore says `scope: newly_generated_only` and
`new_own_inputs: NOT_EXPORTED`; this is not complete sharing of what GLM knows.
A later controller needs a verified own-input delta policy to include new own inputs
without duplicating reconstructed context, and must count that additional channel.

Even with a finished marker, every manifest says `full_completion: false` and
`final_tail: UNKNOWN`; evidence is `EXPORTED_PREFIX` or `EMPTY_EXPORTED_PREFIX`.
A short tool turn may have no generated rows exported, and the last sampled token may
never have been processed into a KV entry at all.
No extra forward pass, final drain, speculative-row export or cache reread is invented.

## Next prerequisite and validation

The next controller step is pinned forward translation of these selected files and
bounded append/acknowledgement through Qwen's separate activation FD, with no model-facing
file path or activation command added to OMP's own-input channel.
Complete-tail evidence needs a separately reviewed connector lifecycle change that
holds request blocks, exports only accepted canonical rows, acknowledges completion
from workers and then permits freeing; it cannot be claimed from this collector.

Local red regressions first failed because the collector was absent and because native
restoration always disabled taps; fixes are covered by synthetic canonical arrays and
fake transports, including generated-only boundaries, missing tail, actual reduced
token limit, stale sessions, malformed/nonfinite data, gaps, errors, caps and deadlines.
Baseline: 585 reference tests passed with three runtime/adapter skips.
Focused command uses `PYTHONPATH=. /opt/drift/.venv/bin/python -m pytest -q tests/test_glm_turn_outbox.py tests/test_glm_restore.py tests/test_glm_restore_factory.py tests/test_glm_snapshot_binding.py tests/test_glm_session.py` (43 passed).
No host inference, serving restart, connector promotion or private activation inspection occurred
in implementation; model inference cost is zero and energy was not measured.
Final checks: 599 reference tests passed / three skipped real/runtime adapter gates
(34.99 s), next-runtime 155 passed / one pre-existing expected sparse-indexer-tie failure
(8.01 s), 59 plugin tests passed, TypeScript compilation and whitespace checks passed.
Skipped gates remain BLOCKED; these tests qualify mechanics only.
