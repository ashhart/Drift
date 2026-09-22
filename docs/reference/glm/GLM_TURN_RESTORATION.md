# Native GLM per-turn snapshot restoration

This adapter is an experimental engineering mechanism, not a live collaboration pass.
`drift.serving.glm_restore_factory.factory` keeps the existing native chat endpoint and
its deployed tool parser; each tool turn reconstructs GLM's own canonical transcript
and restores one explicitly pinned foreign snapshot into a fresh connector session.
The default `glm_factory` remains unchanged and unlinked.

## Recipe and ownership

The owner selects `memory_mode: linked_snapshot` or `no_link_reserved` and supplies the
existing model/translator manifest pins and service authentication configuration.
`restoration` contains `snapshot_path`, `snapshot_sha256`, `rows`, `max_rows`, `layers`
(integer layer indices), distinct `source_worker` and `target_worker` labels, and for
linked mode `route_path` and `route_sha256`.
The snapshot must have precisely those `lN` arrays, each finite float `(rows, 512)`,
within the explicit row budget and one MiB compressed-file limit.
Model/translator manifest verification remains mandatory; metadata pins do not claim
that all weight bytes were independently hashed.
The owner must bind the declared layer set to the loaded checkpoint; the deployed
receiver also rejects missing or unexpected layers before mutation.

The pinned route JSON contains exactly `peer`, `hostname`, `identity_file`,
`known_hosts_file`, and `known_hosts_sha256`.
All artifact paths must be canonical absolute files, and the identity file must have
private permissions; the route uses its explicit identity and known-host file with
strict host checking and ignores global SSH configuration.
The peer hostname is checked before staging; the parent prepares the actual profile.
No adapter construction launches a model request or changes serving flags.

## Native reservation verification

The factory inserts repeated existing `[MASK]` tokens at the beginning of the first
own system message, retaining every original message/tool field and its order.
A separate owned child calls the native `/tokenize` endpoint for original and reserved
messages, requiring an exact insertion of the declared number of token ID 154821,
with every original token before and after that span identical.
An original prompt already containing that marker ID is rejected as ambiguous.
Private token IDs remain in that child; its sole output is verified counts and offset.
The ordinary `GlmHttp.count_tokens` path is unchanged and independently recounts the
reserved request; its count must agree with the proof and native final usage.
Each generation charges the whole reconstructed prefill, including reservation tokens.
Tokenizer proof, all-rank staging and receipt waits share the worker's absolute deadline.

Read-only tokenizer evidence from the deployed native GLM service used a public
synthetic system/user/tool definition: original 156 tokens, reserved 164, eight
verified marker IDs at offset 143, unchanged prefix and suffix.
The pinned native template SHA-256 was
`7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb`.
A request-level template override was rejected by the deployed server; the implementation
therefore does not use an override or assume newline token counts.
No model completion was requested for this evidence.

## Delivery and limits

Every reconstructed request gets a fresh `restore-<uuid>` directory under the existing
`/dev/shm/glm53-handoff` root, a fresh cache salt, and disabled outgoing taps.
Both rank files must match the snapshot digest before dispatch, and native chat events
are withheld until both post-write receipts match session, sequence zero, rows and digest.
The native connector is unchanged; this does not make two-rank application atomic.
Failed receipt collection poisons the session and closes owned HTTP/prefix children;
existing GLM engine-recovery policy remains responsible for server cancellation evidence.
Private snapshot files and receipts are retained for the owner's evidence lifecycle.

The connector applies memory after prefill logits have been computed, so this mechanism
cannot establish first-token causality; aggregate receipts explicitly say
`first_token_causality: NOT_ESTABLISHED`.
The matched control must use `no_link_reserved`, which retains the exact reservation;
plain original-text prompting is a different arm.
Fixed snapshot restoration does not yet refresh partner memory dynamically, prove source
attribution, preserve native GLM engine cache between calls, or qualify two-model coding.
It reports `native_state: reconstructed` through the unchanged worker protocol.

## Verification and next gate

Local regressions first failed for absent adapters, then separately exposed unbounded
prepared-input bytes and symlinked memory parents before their fixes.
Tests cover exact token insertion, real owned-child tokenizer proofs and failure reaping,
fresh turn/session directories, missing-rank suppression, immutable own requests,
matching reserved no-link prompts, fixed deadlines, pinned snapshot/layout validation,
and changed route host-key files.
Checks use the main virtual environments with this isolated checkout on `PYTHONPATH`;
no hidden items, activation contents or generated model text were inspected.
GPU inference cost is zero for this ticket; energy and wall-clock service costs were not
measured, and tokenizer RPC overhead must be recorded in the future matched run.
Next gate: owner-approved single bounded native tool round trip with an independently
pinned snapshot and explicit route, validating actual reservation receipts, tool parsing,
full prefill accounting and owned server cancellation before any Duo comparison.

Final local checks on the isolated checkout: reference 537 passed / 3 skipped (33.55 s),
next-runtime 155 passed / 1 pre-existing expected sparse-indexer-tie failure (7.84 s),
plugin 47 passed, TypeScript compilation passed, and `git diff --check` passed.
Reference command: `PYTHONPATH=. /opt/drift/.venv/bin/python -m pytest -q`.
Focused command uses that interpreter with `tests/test_glm_restore.py`,
`tests/test_glm_restore_factory.py`, `tests/test_glm_restore_prefix.py`,
`tests/test_glm_http.py` and `tests/test_glm_session.py` (34 passed).
Next-runtime and plugin checks follow `scripts/check_all.sh` with the main checkout's
existing interpreter/dependency paths; no packages were installed.
Skipped real adapter gates remain BLOCKED, not passed by these synthetic tests.
