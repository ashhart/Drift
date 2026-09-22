# Controller-owned GLM snapshot versions

This is a bounded local engineering primitive for replacing the foreign snapshot
between reconstructed GLM tool turns; it is not a continuous two-model qualification.
The existing fixed-snapshot factory and worker own-input protocol are unchanged.

`SnapshotBank` in `drift.serving.glm_snapshot_bank` owns a fresh private subdirectory
under an existing canonical mode-0700 controller root.
The constructor binds `session`, distinct `source_worker` and `target_worker`,
`recipe_sha256`, exact GLM `layouts`, `max_rows`, `max_bytes`, `max_versions` and
`max_total_bytes`.
Its hard ceilings are 4,096 rows, one MiB per publication, 1,024 versions and
128 MiB total retained publication bytes; an owner must choose the actual smaller
run limits, and these ceilings do not authorize compute or context capacity.
All versions remain until the controller's separate evidence-retention cleanup.

The trusted controller calls `publish(frame)` with exactly:

```json
{"v":1,"session":"controller-session","version":1,"path":"translated-1.npz","sha256":"<64 hex>","rows":12,"source_worker":"Qwen","target_worker":"GLM","recipe_sha256":"<translator manifest SHA-256>"}
```

`path` is a basename within the controller root, not an arbitrary host path.
The bank copies through a bounded regular-file reader, verifies the expected hash,
exact complete finite layer arrays, row counts and archive expansion bound, then
creates a read-only version file without replacing an existing file.
Versions must increase by exactly one; duplicate, skipped, mislabelled or corrupt
updates poison the bank, as do tampered retained versions.
Both stored-byte and version budgets persist across captures.
The publish receipt contains counters, hashes and ownership metadata, never file
contents, prompt text, token IDs or generated text.

`capture()` returns an immutable version reference and checks its retained bytes;
`validate(reference)` accepts only that bank's own captured references.
Replacing an incoming staging file cannot replace the captured private copy.
The controller may publish while a request is running, but this only changes the
version available for the next preparation; it does not write into the running cache.
This is latest-complete-version selection, not the research scheduler's qualified
one-epoch-lag two-reader protocol.

## Binding to the concrete reader

After creating the existing linked GLM session and publishing its pinned initial
snapshot as bank version one, call
`bind_snapshot_bank(session, bank, route, translator_sha256=...)` from
`drift.serving.glm_snapshot_binding` before `session.open`.
Here `route` is the existing verified `PinnedRoute`; the expected translator manifest
hash comes from the same owner configuration that created the fixed session.
Binding requires an unopened, unpoisoned linked session, matching initial snapshot
digest/rows, matching source/target labels and matching translator manifest hash.
It replaces only the restoration helper behind the existing `GlmSession.prepare_turn`
and `RestoringTransport` paths.

Every prepare captures one complete version under a turn-boundary lock, verifies its
native placeholder span, and uses that captured path/hash/row count throughout staging,
native input counting and rank acknowledgement.
Even an update during prefix verification is deferred to the next turn.
Each aggregate completed-turn receipt includes a `snapshot` record with its version,
bank session, recipe hash, owners, byte count and row count alongside the existing
exact two-rank receipts.
Corruption discovered during an active turn suppresses subsequent reader events and
closes the owned transport; the existing server-recovery contract still applies.
Full reconstructed prefill accounting and the first-token-causality limitation of the
underlying connector remain unchanged.

## Remaining connection and evidence

The current `scripts/live/drift_loop.py` still taps Qwen through the legacy worker
command interface, translates the rows, and pushes numbered files directly to the
continuous GLM connector; it does not use this bank.
The Studio worker's separate activation-control FD supports typed append/tap operations
at generation boundaries, but connecting its publications, translation and acknowledgments
to this controller-owned bank still requires a bounded coordinator.
Likewise, native GLM tool turns currently disable outgoing taps; exposing their own
canonical exports to Qwen's activation FD is a separate missing connection.
No new worker own-input frame, model-facing tool or general network endpoint was added.

Local tests use synthetic arrays and transports only: captured immutability, pending
version selection, all-layer validation, poisoned corruption, byte/version limits,
changed row budgets and exact per-turn metadata/receipt binding.
No model, host, deployed connector, frozen evaluation or backbone was changed.
Source attribution, long-context selection, causal benefit and a paired Duo coding
comparison remain unqualified by this primitive.
The next ticket is the controller connection with explicit resource and ownership
profiles, followed by bounded real two-turn update evidence before simultaneous use.

A regression also changes the caller's update dictionary while its file is being
copied; publication freezes that metadata before I/O, preventing a mismatched version
receipt. A poisoned bank is checked before native dispatch as well as before exposing
reader events. These tests were observed failing before their corresponding fixes.
Baseline reference checks passed 546 tests with three skipped real/runtime gates.
Final focused command is `PYTHONPATH=. /opt/drift/.venv/bin/python -m pytest -q tests/test_glm_snapshot_bank.py tests/test_glm_snapshot_binding.py tests/test_glm_restore.py tests/test_glm_restore_factory.py tests/test_glm_restore_prefix.py`.
It passes 36 tests; full checks use the existing main virtual environments and plugin
dependencies, with this isolated checkout as the import root.
Model inference and remote-operation counts are zero; no energy measurements were made.
Final verification after the fixes: reference 565 passed / 3 skipped (34.95 s),
next-runtime 155 passed / one pre-existing expected sparse-indexer-tie failure (7.77 s),
plugin 59 passed, TypeScript compilation passed and whitespace checks passed.
Skipped adapter gates remain BLOCKED; synthetic success does not upgrade them.
