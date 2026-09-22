# Agent progress

> Historical record, preserved from the pre-cleanup documentation.
> Not current operating instructions; see [current status](../STATUS.md).

This public log keeps engineering status and commands, not private host inventories,
run receipts, local paths or experiment notebooks. Original records are preserved
outside this repository. Do not use this file as evidence of native qualification.

## Composed exchange repair

Code commit: `4dee39b`; base commit: `d317c648f22290beec1787ab787323c58767d9b1`.
Stage: local M4/M5 integration, no scientific promotion.
Changes: normalized typed mailbox messages with separate terminal validation;
reconnecting coordinator with persistent cursors and budgets; opt-in provider exchange
after both actors park, with release bound to the validated receipt digest.

Commands: full `.venv/bin/python -m pytest -q`, focused exchange/mailbox/provider
tests, `bun test`, and `node node_modules/typescript/bin/tsc -p tsconfig.json`.
Red regressions: four mailbox failures, two actual-client failures and one missing
parking callback. After repair: 1252 reference tests passed, three environment skips,
60.19 seconds; 170 plugin tests passed with 655 assertions in 2.32 seconds; TypeScript
passed. Costs were local synthetic tests only; money and energy were not measured.
No pretrained models, hosts or deployments were used. Native qualification is BLOCKED.
The project boundary is one-shot, not a qualified continuous exchange scheduler.

Next ticket: verify ownership, deployment hashes and native runtime qualification
before any live execution. Skipped parity environments are not a scientific pass.

## Repository cleanup

Development receipts, publication drafts and assets, historical build logs and the
obsolete extraction manifest were archived outside the repository. Identifying
paths, hosts and network examples were replaced with explicit placeholders.
Audit reports and source backups stay outside the publication tree. Retained Git
history is not sanitized, and publication remains BLOCKED pending a separate
history/identity decision and final review of the chosen publication snapshot.

Cleanup base: `4dee39bdc0c20fc1593cfbfbfffcf59e13c1e411`.
Post-cleanup QA: 1257 reference tests passed with three environment skips in 60.91
seconds; 173 next-environment tests passed with one known parity xfail in 8.35 seconds;
170 plugin tests passed with 655 assertions, and TypeScript passed.
The focused deployment/profile selection passed 48 tests. No host access or deployment
was performed; private runtime examples must be configured and verified before use.

## Native connector deployment

Source commit: `dc7820146c482c439f34387cd36f196456500a90`.
Stage: M4 deployment prerequisite, no scientific promotion.
The owner confirmed exclusive host availability and approved coordinated restart
and bounded qualification. Both original bundles and launch configurations were
preserved privately before mutation. Both ranks passed complete staged imports,
runtime identity and rollback-inventory checks. Both serving containers were then
stopped before either active bundle changed; all eleven installed module hashes
were verified on both stopped ranks before both containers restarted.
Post-restart file/runtime comparison: PASSED on both ranks.
Cold service recovery: PASSED. Native no-link cancellation: PASSED in 12.73 seconds,
with three idle baseline samples, observed output, explicit scoped abort, reaped
supervisor without escalation, and three idle recovery samples at baseline KV use.
There was one request, 26 own prompt tokens plus eight reserved tokens, a 256-token
generation cap, zero retries and no warm-up; actual generated tokens were not measured.
Cancellation receipt SHA-256:
`866089ccf5d5d71b65fc0e6e3ee050a010695f62ff5a4b69d0e33b21073d920e`.
Two-rank source-check SHA-256:
`e87486ba5fbb3d8649a810c0454879e0f54b9f02fed9932529ccf5392ba2ccc4`.
These checks do not qualify linked cache writes, MCDMA recall, Qwen cleanup, or Duo.

Commands: full `.venv/bin/python -m pytest -q`, focused deployment/cancellation
tests, pinned private deployment preparation and both-rank staged checks.
Baseline: 1257 reference tests passed, three environment skips, 61.54 seconds.
Focused deployment/cancellation regression tests: 49 passed in 1.02 seconds.
No checkpoint, translator, driver, service security or launch configuration changed.
Private audit records retain exact host inventory, commands, receipts and rollback
copies; energy and monetary cost were not measured. Nothing was published.
Next ticket: native linked-cache qualification over MCDMA using the repaired bundle;
the older SSH-payload qualification path was not dispatched as a substitute.

## In-flight exchange cancellation

Code commit: `5d02fa7fd287f93f07b836ba9e16acb63144b292`.
Base commit: `ede37a1afeb620c5ac6dbc896c0f25ce185000c4`.
Stage: local M4/M5 integration prerequisite, no scientific promotion.
The coordinator previously checked cancellation and its deadline only between
requests. Nine deterministic socket cases reproduced publication, append or ACK
after disconnection, expiry or shutdown, with the route still healthy. Two further
cases reproduced continued rank staging and forward reads after cancellation.

Changes: request-scoped lifetime checks surround source/link/sink operations;
mailbox connections check each native read/write; rank-staging threads inherit
the same check; reverse wait budgets use monotonic time. An interrupted route
stays poisoned. Already issued native calls cannot be rolled back or preempted
by this code, so independent native cleanup evidence remains required.
The shipped JS client's actual AbortSignal path now has a composed regression
against the Python coordinator, in addition to the socket and mailbox cases.

Commands: `.venv/bin/python -m pytest -q`, focused exchange/mailbox selection,
the established next-environment parity selection, `bun test`, TypeScript,
`git diff --cached --check` and staged-diff secret scanning.
Baseline: 1257 reference tests passed, three environment skips, 63.25 seconds.
Final reference: PASSED, 1269 tests, three environment skips, 68.74 seconds.
Focused regression: PASSED, 90 tests in 9.47 seconds.
Next environment: 173 passed, one existing sparse-indexer parity xfail, 11.28 seconds.
Plugin: PASSED, 170 tests and 655 assertions in 2.04 seconds; TypeScript PASSED.
Environment skips and the known parity xfail are not native-adapter passes.
Reference log SHA-256:
`11f9e0799d49332ea3d33e0143caa43e093b10b8f98016131ab06d3e02f06d7d`.
Focused log SHA-256:
`542bb5088343c406145b5bfbc9db9096fe23e2cac3d7887212e38bf5d11e5f3d`.

All new checks used synthetic data and local processes; no host, model, deployment,
checkpoint or translator was touched. Monetary and energy costs were not measured.
Native two-way MCDMA/OMP qualification remains BLOCKED. The repository supplies
injectable live adapters but no production runtime bootstrap instantiates the
coordinator against the native parked workers. The exploratory loop remains a
separate path with example transport addresses, not an OMP qualification runner.
Next ticket: bind the native worker-owned snapshot/append controls and verified
private MCDMA configuration to the coordinator before a bounded two-actor run;
require both-direction application, cancellation and independent cleanup receipts.
Nothing was pushed, and the earlier clean-history candidate is now out of date.

## Remaining documentation path cleanup

Stage: repository hygiene, no scientific promotion.
Documentation fix commit: `69986d3e6adf45bf6e5f67b765f4a52d2efa93e1`.
The two native-serialization documents now use a complete example username;
runtime code, tests, model artifacts and deployed hosts are unchanged.
Commands: `.venv/bin/python -m pytest -q`, then after the edits
`.venv/bin/python -m pytest -q tests/test_worker_assistant_prefix.py tests/test_worker_studio_control.py`,
the full reference command again, and `git diff --cached --check`.
Baseline: 1269 passed, three environment skips, 61.97 seconds.
Focused: PASSED, 13 tests, 0.03 seconds.
Final reference: PASSED, 1269 tests, three environment skips, 59.94 seconds.
The skips remain BLOCKED native-adapter gates, not qualification passes.
Focused log SHA-256: `2517e9eae50646d7bbe12680a6b4bab62b845ea0743476059c8b765356d3b1f7`.
Reference log SHA-256: `d420c9322ff77cb041a31186d4aadd3461352c088bb0ccfd24662d4421370b59`.
No training, model inference, native transfers or host operations were performed;
energy and monetary costs were not measured.
Publication preparation and recovery evidence remain outside this repository;
development history remains private and must not be copied into a public repository.
Next ticket: approve the public commit identity and audit the fresh publication
snapshot before any separately authorized upload; native qualification is unchanged.

## README and command entry points

Stage: documentation, no scientific promotion.
README commit: `25b43ec59464b17ef846ee3e55c85b8b4b78916c`.
The entry point now explains the memory channel, local setup, tap creation,
adapter scaffolding, exit codes, workflow integration and the agent contract.
It distinguishes the shell CLI from the OMP slash command and keeps native
startup, two-way recall and main-agent/subagent qualification explicitly open.
Read-only inspection confirmed Duo registers `duo`, while this plugin registers
`drift`; exposing the latter does not require changing Duo's repository.
No Duo files, provider logic, host configuration or native services changed.

Baseline reference: 1269 passed, three environment skips, 68.60 seconds.
Focused `.venv/bin/python -m pytest -q tests/test_tap_cli.py`: 8 passed, 1.89 seconds.
Final `.venv/bin/python -m pytest -q`: 1269 passed, three environment skips,
65.67 seconds; skips remain BLOCKED qualification gates.
README checks: 20 local links, five shell blocks and four parsed CLI examples
PASSED; `git diff --check` PASSED.
The existing environment lacked pip; an offline, no-dependency editable install
with uv supplied the console entry point, and `drift --help` plus `drift adapter list`
ran successfully without loading a model.
The documented random-toy self-handoff PASSED with unchanged weights;
semantic transfer was NOT_EVALUATED.
README SHA-256: `7103b11dd54a976830bb227129aef5d837aced201af2c0b4c4bfeb1110ace66c`.
Reference log SHA-256: `e1e31963569fb3a83bf59dede270397bde09c12775310b7e4e5f2ac216976541`.
README check SHA-256: `ffd9a89698f6b3ddcdec661d5a7ada68bf11f93fec33015eff391092732bc424`.
No real-model inference, training or native transfer ran; energy and currency
were not measured. Publication identity approval and audit records remain private;
nothing was pushed. Next ticket: native coordinator bootstrap and bounded
two-actor qualification before enabling a live Duo text/Drift mode switch.

## README artwork

Stage: documentation, no scientific promotion.
Artwork commit: `b96d6f5eaf34c0252e202706731ecffc53dd06cd`.
Added the supplied PNG below the README title without altering its bytes.
Image SHA-256: `f21f182dae859ba47c2563fcc3da20976156f75814ff94df0422df2d4b2b5d19`.
PNG structure, chunk checksums, decoded provenance metadata and visual inspection
completed; the metadata secret scan found no confirmed secrets.
Private inspection records remain outside this repository.
README checks: PASSED, 21 local links, five shell blocks and four CLI examples.
`git diff --check`: PASSED.
`.venv/bin/python -m pytest -q`: 1269 passed, three environment skips,
64.17 seconds; the skipped native-adapter gates remain BLOCKED.
Reference log SHA-256: `ab7f181cc61484e980d01a1d107a578b206d60899b3eca8ef1e7e6847e811eca`.
Read-only GitHub inspection found no translator weight artifacts in the main
tree or release assets; tap creation consumes translators but does not train them.
No model inference, training, host operations or uploads ran; energy and monetary
costs were not measured. Native qualification is unchanged.
Next ticket: separately authorize publication of the artwork after auditing the
outgoing candidate; native coordinator qualification remains open.

## Deferred native application prerequisite

Stage: M4/M5 coordinator lifecycle prerequisite, no scientific promotion.
Code commit: `aa0fe9284e0d2318b7f33a1ac6881973c2584e44`.
Baseline: `.venv/bin/python -m pytest -q`, 1269 passed, three environment
skips, 68.58 seconds.
The parked GLM worker reconstructs a fresh native request after release, while
the synchronous exchange waits for application before returning. These paths
cannot be connected unchanged without waiting on work that has not started.
This was a missing protocol phase, not a measured native timeout.

Added separate `deliver` and sequence-bound `confirm` operations, retaining
one pending publication without advancing confirmed cursors or row totals.
Delivery records cannot pass the existing application-evidence validator.
The old synchronous operation and provider release checks remain unchanged.
Coordinator termination now poisons its routes, including pending deliveries.
The outbound state is a separate 40-line module; no model or transport API changed.

Red first: `.venv/bin/python -m pytest -q tests/test_exchange_deferred_application.py`
failed ten cases because the coordinator had no deferred operations.
The later owner-lifetime test failed both shutdown and deadline cases before
route invalidation was added. Both regressions now pass.
Focused exchange, guard, coordinator, cancellation and real JS-client checks:
PASSED, 61 tests, 0.53 seconds.
Final `.venv/bin/python -m pytest -q`: 1285 passed, three environment skips,
62.50 seconds; skips remain BLOCKED native-adapter qualification gates.
`bun test` in `plugin/omp-drift`: 170 passed, 655 assertions, 2.43 seconds.
`bunx tsc --noEmit` and `git diff --check`: PASSED.
Focused log SHA-256: `2aee7dbf2b5d469cae7d239e0fc89dd58bfd69b4400a2b65868c1652d5f48465`.
Reference log SHA-256: `e8b92ede767a45e44c2bef9d6b05d2180976a206addcfe3819611368d32e6e06`.
Plugin log SHA-256: `dd36544a2756d842d949c9bfd901bbb6fb7a295e11a4c070a0deeb3fe8dd20b1`.

Read-only host discovery confirmed all three hosts reachable and the dedicated
targets present on both ranks; no native model, bridge or deployment was started.
The owner scoped this release to GLM/Qwen and approved ten minutes of native
inference without training or shared-daemon restarts; none was consumed here.
Energy and monetary costs were not measured. Changed-content secret scanning
found no confirmed findings, but this is not a full outgoing-publication audit.
Nothing was pushed and no release or scientific gate was promoted.

Next ticket: bind a runtime-owned coordinator to the versioned GLM snapshot bank
and fresh request identities, then qualify resume/application and cleanup.
Native `/drift` startup, repeating tool/subagent scheduling, new own-input export,
recall, source ownership, final-tail coverage and two-host CLI qualification remain
open; existing translator artifacts are still required by the tap CLI.

### 2026-09-21: README banner published; translator distribution inventory

Stage: documentation publication PASSED; translator release BLOCKED pending
artifact provenance and exact model compatibility review, with no native gate change.
Published commit: `3ddf46baed5018edf33b6d2f50703d3d831d388a`, from the sanitized
publication checkout, parent `3b59520335fbfc473a5ced58dec2a07668d57132`.
Only README.md and docs/assets/drift.png changed; private development history
and unqualified integration changes were not pushed.
Commands: candidate audit, Gitleaks tree/object/history scans, README checker,
scoped `git push origin main:refs/heads/main`, `git ls-remote`, GitHub contents API.
Checks: 21 links, five bash blocks and four CLI examples PASSED; the identical
banner change's previously recorded reference result remains 1269 passed and
three BLOCKED environment skips. No code changed or native compute was used.
Banner SHA-256: `f21f182dae859ba47c2563fcc3da20976156f75814ff94df0422df2d4b2b5d19`.
README SHA-256: `16ebbb4b4b61a62e86fb0268a5e54b18ea3eda9e688287d601ccd68100cfb8d4`.
No confirmed new privacy finding; prior narrow scanner dispositions and the
supplied image's reviewed C2PA provenance were retained, not blanket-allowlisted.
Remote main and banner blob were verified after pushing; the deny hook remains.
Located the five live GLM/Qwen translator artifacts and verified four against
frozen manifests, with numeric NPZ arrays finite and no pickle loading.
The live format differs from the CLI's pool.v1 files, which explicitly record
semantic_transfer=NOT_EVALUATED; none of these weights was uploaded.
Next ticket: verify backbone revisions and redistribution provenance, then ship
the appropriate versioned translator bundle with checksums and usable instructions.
Energy and monetary costs were not measured; no training or inference ran.

### 2026-09-21: Runtime-owned MCDMA restoration transport

Stage: local restoration mechanics PASSED; native coordinator and OMP
qualification remain BLOCKED, with no research-stage advancement.
Code commit: `a0177d0a527a0cafd9c5bd78fb42ea8aa07e27be`.

The GLM snapshot bank previously reconstructed each turn through an SSH-only
publication transport. Added a runtime-supplied MCDMA restoration factory and
carried it through the initial factory, versioned bank and owner serving path.
The declared transport must match the supplied factory; no MCDMA-to-SSH fallback.
Separated all-rank staging from head release without weakening application
receipts. Partial staging, wrong transaction ownership, invalid timeouts,
missing ranks and mismatched applied digests fail closed. Nested transport
checks retain the outer request's cancellation check.

Red first: `.venv/bin/python -m pytest -q tests/test_mcdma_restoration.py`
failed four tests for absent APIs. The real startup parser still reports the
intentional reference-only BLOCKED guard; it was not bypassed.
Composed tests use real mailbox framing, Spark bridge and GLM restoration
wrappers but synthetic regions and generation, across two snapshot versions.
Later receipt regression tests first failed four cases and then passed.
The first full run caught a None-checker cancellation regression; it was fixed.

Baseline `.venv/bin/python -m pytest -q`: 1285 passed, three skips, 61.48 seconds.
Final focused restoration, bank, owner, coordinator, mailbox and cancellation
tests: 86 passed, 10.35 seconds.
Final `.venv/bin/python -m pytest -q`: 1307 passed, three environment skips,
59.30 seconds; skips remain BLOCKED real-adapter qualifications.
`bun test`: 170 passed, 655 assertions, 2.37 seconds.
`bunx tsc --noEmit` and `git diff --check`: PASSED.
Focused log SHA-256: `ef7ea7936a0319c18fcd23722cc99aa8cb4e37b172c5d7e90e6b531f38560759`.
Reference log SHA-256: `3a0cb35eb727686e66a99453815a1e4e16cb8f299a8cbf28a1b1102a34febd75`.
Plugin log SHA-256: `e1c67db23c5b255b676642cd36b83e33c5823b7636ad1d6c00643c6f08faece7`.

Hosts were inspected read-only; no deployments, region writes, restarts or
native inference ran. The approved ten-minute inference allowance is unused.
Energy and monetary costs were not measured. Changed-content secret scanning
found no findings; a full publication audit is still required for a release.
These functional changes were not pushed; the prior README push is separate.

Next ticket: runtime-owned native bootstrap with pinned connections and the
forward mailbox, followed by the provider's delivery/resume/application schedule.
Native startup, repeated OMP/subagent exchanges, complete own-input and final-tail
export, recall/source ownership and native cleanup evidence are still open.

## 2026-09-22: native connection admission and successive forward requests

Stage: runtime-owned native bootstrap prerequisite, PASSED locally; native
startup and scientific qualification remain BLOCKED.
Code commit: `5ea727ece82c4b9d9aef2126d810160a4e2cf817`.
Concurrent exchange-seam repair `201416d262b98a23bde011e30969ecff0613b4c5`
landed before the final QA run and was preserved.

The bridge rejected a second native request even after acknowledging the first
request's complete forward stream. The four-test reproduction failed two cases
before repair. Request changes now retain the writer and wire sequence, require
the terminal ACK and reject old identities, unfinished streams and failed writers.
Connection admission verifies both regions and the separate forward connection
before stamping its window, closes every acquired connection on partial failure
and refuses operations after closure or its absolute deadline. Twelve new tests
first failed for the absent connection-owner API, then passed with that API.
No driver, daemon, model or remote region was touched.

The README now links the verified public MCDMA repository and setup guide without
claiming that installing the transport enables the unfinished native workflow.
The GitHub skill also caught private hostname fixtures introduced by the concurrent
commit; they were replaced with fictional rank labels without changing assertions.
Original development history remains private and must not be pushed.

Baseline `.venv/bin/python -m pytest -q`: 1307 passed, three skips, 60.83 seconds.
Focused mailbox, restoration and connection tests: 72 passed, 9.15 seconds.
Sanitized exchange-seam and new connection/turn tests: 24 passed, 0.05 seconds.
Final `.venv/bin/python -m pytest -q`: 1331 passed, three environment skips,
59.01 seconds, one existing training warning; skips remain BLOCKED native gates.
An intermediate full run hit an import error while another process was editing
the exchange seam; it is not used as release evidence.
`git diff --check` and `bunx tsc --noEmit`: PASSED.
Focused log SHA-256: `99290dafeae5224966cfa4169fab2ff11c32eadfd0490f3ff41a9b6eb699ef8b`.
Sanitized focused SHA-256: `cafad75a79093740238130e07207fe460b41db8a4f60deb3955765fccd7138fc`.
Final reference SHA-256: `47316f0180f5fff56dc6fd4f46f088e3c1176eb0755dd3871bad3dc40ba9edab`.
`bun test`: 170 passed, 655 assertions, 2.37 seconds.
Plugin log SHA-256: `e122871d7bfc6bd84279c27e80e32a8041705934b51173c17cafdb0a5fa82f13`.

No native inference, training, deployment or shared-daemon restart ran; the
ten-minute inference allowance remains unused. Energy and money were not measured.
Native host availability must be coordinated again after concurrent host work.
Publication remains blocked pending functional completion and a frozen final
candidate audit; source repairs and README changes in this ticket were not pushed.
Next ticket: connect the owned transport to the GLM forward collector and worker
bootstrap, then the repeating provider schedule and bounded native recall proof.

## 2026-09-22: native worker forward mailbox collector

Stage: runtime-owned forward collection PASSED locally; native pair startup and
scientific qualification remain BLOCKED, with no research-stage advancement.
Code commit: `aa2c7a04819e4174f025e677df8bce8e9764cb1b`.

Added the focused McdmaTurnOutbox module and runtime-only forward-factory wiring
through restoration, snapshot-bank ownership and the worker entry point.
The collector watches each fresh request, retains the transport sequence across
turns, validates the source frontier and capacity before acknowledgement, and
captures immutable generated-row files before freeing the mailbox slot.
Completion acknowledgement follows manifest creation; cancellation, invalid
input, deadline expiry and failed local writes poison the collector.
The worker's remaining absolute deadline also bounds its watch operation.
No SSH activation fallback or new model-supplied import path was introduced.

Eleven initial collector tests failed for the missing module, then passed.
Five owner-factory tests failed for the absent callback, then passed.
Further tests cover cancellation and invalid deadlines; a composed test runs
two snapshot versions through the actual bridge, mailbox and restoration code,
with synthetic region IO and generation. Mailbox capture is explicitly not
Qwen cache application; new own-input selection and native final-tail evidence
remain separate gaps.

Baseline `.venv/bin/python -m pytest -q`: 1331 passed, three skips, 61.35 seconds.
Focused `.venv/bin/python -m pytest -q tests/test_glm_restore_factory.py
tests/test_mcdma_turn_outbox.py tests/test_mcdma_restoration.py
tests/test_glm_restore.py tests/test_glm_owner_worker.py
tests/test_glm_snapshot_binding.py`: 75 passed, 1.46 seconds.
Final `.venv/bin/python -m pytest -q`: 1352 passed, three environment skips,
62.99 seconds, one existing training warning; skips remain BLOCKED native gates.
`bun test`: 170 passed, 655 assertions, 2.49 seconds.
`bunx tsc --noEmit` and `git diff --check`: PASSED.
Focused log SHA-256: `b61caf39e9c6a3b90b2e3ffe0dd3ab0f7fb6a9e77316a88ffdfb32759005ecec`.
Reference log SHA-256: `de48cebeb33ba10b00e84696ef5d87d3dc896210fd90b9c9ca4fde71ce60c79d`.
Plugin log SHA-256: `cdf63be00595475b8fea3ba7f603c8abdd6d1056ed0f1683b2b56543e9c839f0`.

The user reconfirmed host ownership. Read-only checks found that all eleven
deployed connector file hashes on both ranks match the source bundle, but the
installed bridge/mailbox files are older. The current native service declares
cache_dtype=auto rather than the reader's qualified fp8_ds_mla contract.
No assumption about actual cache tensor layout or parity was substituted for
that mismatch. Exact private deployment evidence remains outside this repository.
File-comparison evidence SHA-256: `8fa780ffc881f6ed073a4d035ddf72e44e8ba23a266f086947f5a089dc2e3de9`.

No native inference, training, deployment or shared-service restart ran; the
ten-minute inference allowance remains unused. Energy and money were unmeasured.
Changed-content secret scanning covered 28039 bytes without findings; contextual
matches were Python decorators, not email addresses. This is not a full release
audit or permission to push private development history; no source was pushed.

Next ticket: resolve the native cache configuration under explicit restart
authority, stage the current bridge, and finish the Qwen/GLM coordinator owner
and repeated provider schedule before the bounded native recall proof.

## 2026-09-22: authorized native cache configuration restart

Stage: cache configuration restoration PASSED; native recall remains BLOCKED.
Source commit: `65ee073c1b5aa3a6cd1eba6720717b4e0ced8691`.
Collector commit: `aa2c7a04819e4174f025e677df8bce8e9764cb1b`.
The user explicitly authorized a coordinated GLM restart. Both ranks were idle
before shutdown. Saved original launch scripts and environment configuration
outside this repository and on their respective hosts for rollback.
Changed only the main KV cache selection from auto to fp8_ds_mla, retained the
existing containers, images, model revisions and other arguments, and restarted
both GLM ranks. The draft cache setting is unchanged; runtime ablation stays off.
Both launch scripts and the persistent Drift environment file pass `bash -n`.
Both MCDMA targets and both handoff daemons retain their original PIDs and start
times. All eleven deployed connector file hashes still match the manifest on
both ranks; private comparison evidence SHA-256:
`8fa780ffc881f6ed073a4d035ddf72e44e8ba23a266f086947f5a089dc2e3de9`.

Focused `.venv/bin/python -m pytest -q tests/test_vllm_connector.py
tests/test_mcdma_turn_outbox.py tests/test_glm_owner_worker.py`: 28 passed, 1.18 s.
Full `.venv/bin/python -m pytest -q`: 1352 passed, three environment skips,
one existing warning, 63.39 s; skipped native adapter gates remain BLOCKED.
No project implementation changed, training ran, or task inference was sent.
Model initialization and warmup belong to the authorized restart; energy and
money were not measured. No source was pushed.

Readiness PASSED: `/health` returned HTTP 200, metrics report fp8_ds_mla with
zero running and queued requests, and both ranks registered 72 cache layers.
Container initialization took approximately 197 seconds, with about 224 seconds
from stop initiation to HTTP readiness. No task inference was requested; the
ten-minute qualification allowance remains unused. Private readiness evidence:
head SHA-256 `894b8bdb2761c78d846a3a52ec8f9f1727e75f7d747e3b11aabb52a6cf3f86d3`,
worker SHA-256 `89eae72f99cf62af9b4f1803f33d60f31ba17eeea5aff37d7d585a2a426876a6`.
This is configuration and startup evidence, not physical cache parity, two-way
recall, or cancellation cleanup qualification. The existing engine used forced
child cleanup during shutdown; that observation is not a passing cleanup gate.
Next ticket: stage the current bridge and finish runtime bootstrap before the
bounded native recall proof; new-input selection and repeated provider exchange
still need implementation and evidence.

## 2026-09-22: native GLM MCDMA owner entry point and terminal failure

Stage: owner launch mechanics PASSED locally; native two-turn mechanics FAILED
at terminal coverage; recall and repeated agent qualification remain BLOCKED.
Code commit: `77f3a0df8af5162fd853e6c95dcc8a8315ed67a4`.

Added focused modules for pinned native binding/library loading, connection
lifetime ownership, and the GLM owner's command-line entry point. Real restoration
and forward-collector factories now reach the owner without an SSH activation
fallback. The loader checks both artifacts before execution and disables library
search. The operator must supply reviewed native code and exclusive target use.
Process supervision remains required for hard cancellation of native calls.

Red first: twelve missing-runtime tests, ten missing-loader tests, then a failing
remote-HTTP admission test. Added CLI pinning and real owner-protocol composition
coverage. The remote-HTTP case now uses the existing loopback guard before
transport admission; that guard was not relaxed.
Baseline `.venv/bin/python -m pytest -q`: 1352 passed, three skips, 64.08 s.
Focused runtime, loader, CLI, restoration, collector and owner suite: 71 passed,
1.42 s. Final full suite: 1380 passed, three environment skips, one existing
warning, 62.36 s. Skips remain BLOCKED native adapter gates.
Focused log SHA-256 `ae2122c8f8e37c84b9ad43781edcb8f58f8a296ab3a5522f21f36f7c933d1a50`.
Full log SHA-256 `d615755a12418c7e7c4cc2ac6854c8fec91cd7a67c97961b2b16ceee444e04d6`.

Staged repaired bridges into isolated directories and verified their hashes.
Both bounded bridge processes exited; shared MCDMA/handoff daemons kept their
PIDs. No GLM restart or backbone change occurred. The native probe used synthetic
two-row snapshots, not translated Qwen memory, and Qwen was not loaded.
Both GLM ranks confirmed the expected snapshot digest through MCDMA; forward
taps arrived, but the connector wrote a failed terminal-coverage marker.
The bridge rejected it and the owner timed out without a successful first turn,
so no second turn ran. No recall accuracy or source-ownership claim follows.
The exact finalizer exception and request counters were not retained by the
existing error handler; speculative/asynchronous frontier mismatch is a candidate,
not an established cause. The acceptance guard remains unchanged.

Native probe wall time: 90.433 s; independent supervisor reported the child
reaped and no surviving process group, without forced termination. Final service
health returned HTTP 200 with zero running and waiting requests. Conservatively
charge 91 s against the ten-minute allowance, leaving 509 s. No additional native
inference ran after the poisoned session. Energy and money were not measured.
Summary SHA-256 `d36fde403f0d7dbf5c9531f4dd9210343ff8622c0709cd4ea21d1ea88443716e`.
Cleanup SHA-256 `3a296079d26500d7c3c55580e118221873c7afb7a042770775f687e7e834f6f3`.
Private evidence and host details remain outside agent-readable worktrees.

Changed-content secret scanning covered 20686 bytes without findings; this is
not a full publication audit. No source was pushed and private development
history must not be published. Native code staged for the run predates only the
final early HTTP preflight check, not a changed cache or transport contract.
Next ticket: obtain scalar terminal diagnostics under separately scoped GLM
restart authority, fix the measured final-tail failure, then resume native
bootstrap and repeated provider qualification within the remaining allowance.

## 2026-09-22: processed cache frontier repair and native two-turn mechanics

Stage: M-1 native owner mechanics PASSED for two synthetic-memory turns, repeated
twice; cross-family recall and repeated OMP/subagent qualification remain BLOCKED.
Code commit: `0cb207e4bfd823b365472ddd3b0bba952dc2beef`.

The authorized diagnostic restart reproduced the terminal failure and retained
only scalar counters privately. The scheduler reported 39 computed tokens, 31
accepted tokens, eight in-flight tokens and zero stale output tokens. Its
computed cursor includes scheduled work whose result has not yet been processed.
The old finalizer refused any in-flight work, and the worker capture incorrectly
treated the scheduled cursor as verified. Installed scheduler source confirms
processed cache accounting subtracts in-flight tokens.

The connector now retains the scheduler request locally and copies a separate
numeric processed frontier into worker metadata. Publications cannot cross that
frontier. Successful completion trims the existing private candidate to the
minimum of processed and accepted positions; it never rereads freed device pages.
Stale output, invalid counters, rollback, corruption and missing coverage still
fail closed. No task text or token IDs were added to activation transport.

Red first: all six new asynchronous connector regressions failed. Focused
connector/frontier/allocation checks passed 51 tests in 0.31 s. Full baseline:
1380 passed, three environment skips, 63.27 s. Full repaired suite:
1386 passed, three environment skips, one existing warning, 65.28 s.
Final frontier/deployment checks: 43 passed in 0.30 s after correcting a test
filename typo that had collected no tests. Skips remain BLOCKED runtime gates.

Both existing GLM containers were restarted for diagnostics, then restarted with
the repaired leaves and no diagnostic instrumentation. Main-cache fp8_ds_mla,
images and backbone parameters were unchanged. The read-only deployment
comparator PASSED all eleven module hashes and runtime/container identities on
both ranks. Shared MCDMA and handoff daemons retained their PIDs.
Deployment receipt SHA-256:
`4cb399f149be416b786f94ba27fe62bd499d0c74a0aaba545b958d0660b723a7`.

The original two-turn native probe PASSED twice with fresh sessions and target
leases. Each run restored synthetic snapshot versions one and two on both GLM
ranks, completed both forward streams, selected 12 then four generated cache
rows, and reaped the owner process without forced termination or surviving
children. Runs took 2.604 s and 2.089 s. This is transport/lifecycle evidence;
Qwen was not loaded, translators were not invoked, and recall was NOT TESTED.
The last sampled token is not guaranteed to have a cache row, so final-tail
and complete sampled-token coverage claims remain unchanged.
Summary SHA-256 values:
`86d54788cff83b6002535c4c90b615ce5d712bb058b10a959da81bab742ba9d7`,
`336805c336f9f1591455bfe9411ac63c07e01df6868a57de84a36cfed2c21b90`.
Cleanup SHA-256 values:
`61029c7dcd9f346dc1d6d3c5e9d5c83df1aecc66aa997bb04e0cbf36eed8bcd6`,
`faffd04b9f7e861c98caebf3096aaaf11ec28501dd683628a2434addf2b3b269`.

The failed diagnostic consumed 20.405 s. Conservatively charge 27 s for this
turn's three native probes, leaving 482 s of the original inference allowance.
Initialization/warmup belongs to the authorized restarts; no training ran and
energy/money were not measured. Bounded bridges exited; both GLM ranks remain
running and final metrics show zero running or waiting requests.

Changed-content secret checks and whitespace checks passed; these are not a
complete publication audit. No source was pushed. Private traces, activation
files, host details and original history remain outside publication scope.
Next ticket: compose the Qwen owner and pinned forward translation with this
qualified GLM owner, then establish native reciprocal recall and repeating
provider/subagent exchange without claiming this synthetic probe proves them.

## 2026-09-22: native pair composition and failed reciprocal recall

Stage: M-1 native pair exchange mechanics PASSED in bounded exploratory probes;
reciprocal recall FAILED; repeated OMP/subagent qualification remains BLOCKED.
Code commit: `916a0e31357d6c2c95eca02d7cb70aed9ecc93c9`.

The first composed collector-to-translator test failed before inference: the
pinned forward bridge accepted only the historical file-export manifest.
The MCDMA decoder now records the raw payload SHA-256, its collector retains
the terminal frontier, and the bridge verifies the contiguous receipt chain
under a distinct strict MCDMA schema. Capture remains `cache_applied: false`,
`final_tail: UNKNOWN`; the change does not promote scientific evidence.

Baseline: 1386 passed, three environment skips, 66.78 s. The initial composed
test failed with six negative checks passing. After repair, focused checks
passed 87 tests in 0.28 s and the full suite passed 1393 in 61.86 s. Four more
negative cases reject absent raw digests, terminal count/frontier mismatches
and unsupported verified-tail claims. Final focused command:
`.venv/bin/python -m pytest -q tests/test_mcdma_translation_bridge.py tests/test_mcdma_turn_outbox.py tests/test_translation_bridge.py`
passed 57 in 0.31 s. Final full `.venv/bin/python -m pytest -q` passed 1397,
three environment skips and one existing warning in 67.26 s; skips stay BLOCKED.

Three fresh native pair sessions used real frozen models and translators with
no peer text on the activation channel. Each retained two GLM turns, both-rank
application receipts, a forward MCDMA stream and real Qwen cache appends.
The selected-row recipe sent twelve copies of Qwen's final own row and eight
GLM generated rows. Reciprocal recall failed in both directions in the first
run (18.616 s) and a fresh attribution-neutral-question run (15.971 s).
The GLM source's generated answer contained its local fact, so its absence from
the generated source text does not explain the forward miss.

A separate diagnostic translated all 56 Qwen own rows once with the same frozen
translator. Both GLM ranks applied that memory, but GLM still missed recall;
Qwen answered correctly (16.663 s). Selection and copy count both changed, so
this is not a causal isolation or a qualified replacement recipe. No production
selection policy was changed. These are single-case engineering probes without
matched no-link/wrong-memory controls, not benchmark scores or proof that the
translator alone is responsible. Generated-only GLM selection, omitted prompt/
tool rows, reader context/positioning and the attention-copy policy still need
controlled isolation before native reciprocal recall can pass.

Summary SHA-256 values, in run order:
`73a039fb51284f5250cc2a01ec9a88055de6d0d63cabade4f4f6a077a7c098cf`,
`1c2c5d35f524aa8f0998059f37d9192eceabd23d4dc8a5702300543c52e62a68`,
`f3172092b4688848f6f05923966661f72ec041a8786a7fff85bb739060b41fa0`.
Parent cleanup SHA-256 values:
`2f43bebf4413dc0b008b8f74a3edd4e0347719fd4d28ead930df5f855f837811`,
`2ef909a93048bdf7925b60860d4f6ac98f9b0681b617d24cf72f7a841c73dc2a`,
`98a00e6182b875166596aa017883e4e8f920b653ec55a227bf0ae74c941c5f18`.

Two preceding startup attempts are INVALID evidence: one lacked the model
runtime imports (1.901 s), and one closed the outer supervisor's stdin early
(2.022 s parent lifetime; nested worker lifetime not independently measured).
Charge the first 2 s and conservatively charge the second its complete 150 s
outer allowance, plus 19, 16 and 17 s for the three completed probes. Total
charged this turn: 204 s; 278 s remain of the original inference allowance.
No training, backbone modification, server restart or shared-daemon changes
occurred; money and energy were not measured.

All three completed probes reaped their supervised model workers without forced
termination or surviving process groups. Final read-only checks found no owned
probe workers or bridges, zero running/waiting GLM requests and unchanged shared
daemon PIDs. Successful close does not establish cancellation resource release.
Private prompts, responses, activations, host details and diagnosis scripts stay
outside the repository. The repeated provider boundary was not advanced past
this failing native recall gate.

GitHub skill gate: DO NOT PUSH. Staged-code Gitleaks and whitespace checks passed,
but these are not a full publication audit. The sanitized publication checkout
was untouched and original development history remains excluded. No upload ran.
Next ticket: isolate full-source coverage and reader placement/copy policy with
bounded matched controls, then qualify recall before repeating OMP exchanges.

## 2026-09-22 — Guard initial prefill and repeat provider boundaries

Stage: M-1/native integration repair; no scientific stage promotion.
Implementation: `ea9e17e0e3326e706b8399c14d08b312b0c29b53`.
Engineering regression PASSED; general reciprocal recall, provenance, final-tail
coverage and installed-OMP repeating native qualification remain BLOCKED.

The same captured Qwen memory failed under the old system-message layout at one
and twelve copies, but succeeded under aligned prefix layouts at both counts;
the no-link control failed. Native-chat padding likewise passed both linked
variants, with a failing no-link control. The old first prefill could compute
the question before applying memory: later rank receipts did not prove causal
use. A focused scheduler subclass now caps explicit guarded requests before the
own-input suffix; the connector rejects absent or late initial memory. Local
padding is tokenizer-verified and counted. A composed regression caught the
snapshot bank dropping this policy; it now survives rebinding across versions.
Reports distinguish GUARDED_REQUESTED from established first-token causality.

A pinned all-new-own-row, single-copy reverse policy now preserves full tap
intervals and checks row/byte limits before translation. No weights changed.
One authorized coordinated GLM restart deployed the guard and scheduler on both
ranks, preserving supported cache settings and containers. The 13-module
deployment comparison passed; shared transport daemons were untouched.
The final diagnostic passed four linked cases (short/long own input crossed
with one/twelve copies), versus zero of two no-link controls, with both-rank
receipts. All six reused one captured fact, not a held-out benchmark.

A subsequent fresh actual Qwen/GLM pair passed its single reciprocal recall
case: 56 actual Qwen rows translated once and eight generated GLM rows reached
their partners. Both GLM turns applied on both ranks. Both workers exited with
no surviving process groups or forced termination; peer text on the activation
channel was zero. This is not an OMP subagent run, a wrong-memory controlled
study, provenance proof, full row coverage or cancellation qualification.

The opt-in version-2 provider gate has bounded epochs, fresh readiness/exchange/
release records, replay rejection and the original absolute deadline. Only the
initial parent task launch bypasses parking; later task launches exchange.
Production-hook composition tests cover two exchanges then both actors cancelling
during a third. Exchanges are synthetic; native repeating OMP scheduling and its
external release controller remain unqualified. Version 1 stays compatible.

Tests were red before the guard, full-row policy, bank-policy and repeated-task
fixes. Final `.venv/bin/python -m pytest -q`: 1410 passed, three environment skips
(BLOCKED real-adapter gates), one existing warning, 61.24 s. Final `bun test` in
`plugin/omp-drift`: 173 passed, 717 assertions, 2.07 s; `bunx tsc --noEmit` passed.
Bank/prefill/restoration subset: 26 passed; repeating provider subset: three
passed, 62 assertions. Evidence stays outside agent-readable repositories.

Summary SHA-256 values: layout/copy
`821bfceec2fecf09e7d6930d86c52daf586712aa840c5512b5f1c778c6eacb74`;
padded chat `f95a9bf373bfc37625d31ae652174d0848afc4b7ee4c19aa19437f2887bdf9a7`;
guarded six-arm `bd72babad112e82afb2be724ff067f5e93891a2ac22bc804e55b5b95cfbf70af`;
fresh reciprocal `36912e28f1eab77bacd4e3ff1f140b639645a32625837faee1e1ba1e1a90c0bc`.
Reciprocal cleanup: `1c723425d3f4a5fa2fef8dfe62c44e65aadbb7959dafae0fe36d127132552a8a`.
QA logs: Python `b6f9ff4bfe7c93659cf0ab527415190cfdf5c7ea378cd303d0c5942fbe8cf6a9`;
plugin `bad79630c49ec804e61a8aae5ac3e14b169937e7fda023ff9f99e5ffe235663c`.

Native elapsed times: 5.2241, 3.7641, 8.8356 and 18.0427 s. An intervening
4.1597 s startup was refused by the unchanged Studio memory guard and is INVALID
recall evidence. The unrelated workload later released memory without our
intervention, allowing the successful fresh pair. Conservatively charge
6 + 4 + 9 + 19 + 5 = 43 s this turn, leaving 235 s of the original allowance.
Restart initialization is separate; no training ran, and money/energy were not
measured. Private caches, prompts, answers, host details and scripts stay private.

GitHub skill verdict: DO NOT PUSH. Staged-code Gitleaks and whitespace checks
passed, but native repeating qualification and the complete publication audit
remain unfinished. The sanitized publication checkout is unchanged; development
history must never be pushed. Next ticket: connect the repeating provider
lifecycle to the runtime-owned coordinator, then qualify multiple real subagent
epochs, matched recall controls and cancellation within the remaining allowance.

## 2026-09-22: parked native owner staging and startup failure propagation

Stage: repeating owner/provider mechanics PASSED locally; native repetition
FAILED at peer startup and remains BLOCKED. Scientific stages are unchanged.
Implementation commit: `8a09220c851d1a3f2227ada2ba5d14ba2d1987b1`, based on
`97f28d9c00790479bfabb4a3ac87bd69e1014223`.

The first failing integration waited for GLM cache application while the
provider still held GLM at its tool boundary. A shipped-hook regression went
red on that sequence. The new explicit stage operation queues a versioned
snapshot without claiming wire delivery or application. The next boundary
confirms the prior publication before staging another. Completed-turn owner
queries expose only version-bound, both-rank receipts, never prompts or answers.
The final two-actor barrier confirms outstanding publications before teardown.

Focused owner modules compose the real snapshot bank, Qwen activation owner,
MCDMA collector and frozen translators through a private coordinator profile.
The current selected reverse policy is last-own-row with twelve copies; it is
not the full-own-row policy from the earlier reciprocal success. Source reports
retain selected intervals; exchange row counts describe translated rows.
No backbone or translator weights changed, and no training ran.

Three bounded installed-OMP attempts followed. The first failed before remote
startup because the qualification sandbox denied SSH access to the null device.
The second failed configuration validation before inference because its output
token limit exceeded its forward row budget. Both are INVALID recall evidence.
After those fixture corrections, the GLM parent launched the constrained native
task subagent, but the Qwen worker failed before output and the parent parked
until its deadline. No epoch was released. The supervisor reaped the parent;
its orderly owner report was empty, so graceful native cleanup did not pass.
The child's suppressed stderr does not establish its precise startup cause.

That run exposed a provider defect: failures before the first dispatch hook did
not poison the project boundary, leaving the other actor parked. A regression
failed before the fix and passes now. The provider invokes the boundary's abort
on every failure, including startup. This repair is locally tested, not yet
native-qualified. Further negative tests reject wrong-session appends, boolean
numeric receipts, forged confirmations, missing ranks and aborted finalization.

The subsequent read-only Studio preflight found about 123 GiB reclaimable
against the unchanged 198 GiB model-plus-reserve requirement, while another
workload remained resident. No further inference was attempted. No unrelated
process was stopped, no guard weakened and no shared daemon restarted.
Owned control tunnels were closed; bounded bridges and coordinators exited.
The existing dedicated target and handoff daemon process identities were
unchanged on both ranks. Native repetition and linked cancellation remain open.

Commands: `.venv/bin/python -m pytest -q` yielded 1416 passed, three environment
skips, one existing warning in 62.29 s. The skips remain BLOCKED real-adapter
gates. `bun test` in `plugin/omp-drift` yielded 179 passed, 746 assertions in
2.61 s; `bunx tsc --noEmit` passed. Focused Python owner/deferred tests: 32
passed; focused plugin staging/boundary tests: 17 passed, 134 assertions.
An initial focused invocation used a nonexistent test filename and ran no
tests; the corrected commands above passed. `git diff --check` passed.

Evidence SHA-256: failed native summary
`036928f70d7ca4b09d55254857d622f143229ffb76d237d14f29c53b74112307`;
final Python log `81a5ea33b3fbbb98a0d4c6d8f0eef3bcf7a4d34e2e72845eff1f9ad896973214`;
final plugin log `f00865e029e9d4818e8be81615df35d9f14bec1fe6f281b0cdf79921a82cf7f6`.
Evidence, runtime profiles and native outputs remain outside model-readable
repositories. The controlled OMP fixture used 141 counted bytes of static task
control text, no semantic peer handover; it is not stock Duo qualification.

Native wall times: 0.6218, 1.194 and 54.0854 s. Conservatively charge
1 + 2 + 55 = 58 s, leaving 177 s of the original 600 s allowance. The first
two starts did not infer. Money and energy were not measured.

GitHub skill verdict: DO NOT PUSH. Staged-code Gitleaks with recursive decoding,
contextual identifier review and whitespace checks found no confirmed disclosure
in this change; that is not a complete publication audit. The sanitized public
checkout remains unchanged at `3ddf46baed5018edf33b6d2f50703d3d831d388a`.
No GitHub content was sent and development history must never be pushed.
Next ticket: with sufficient native memory, rerun the pinned repeating owner
probe with child stderr captured privately, verify every epoch and independent
cleanup, then complete the full sanitized-candidate audit before any push.

## 2026-09-22 — Explicit subagent completion and terminal capture follow-up

Stage: native provider-boundary repair, not a new scientific milestone.
Implementation commit: `5fe939ba2b68b0fa11f454fa1833b7f4da95994d`.
The preceding terminal-capture repair is `ef0b8c5`; its deployment passed
both-rank module checks before these attempts. Its late-worker regression
reproduced a final tap being overwritten after the completion marker; capture
and finalization now hold the same per-request cross-process lock.

Local completion gate: PASSED. A child text-only stop is not an OMP task
completion. The provider now carries cloned, bounded tool-call arguments into
the gate; explicit terminal yield confirms both pending publications, while
incremental yield remains an exchange boundary. Cancellation and malformed
yield arguments still poison the route. The isolated module-copy closure now
includes the new terminal classifier. No inference timeout was increased.

Red-first commands: `bun test test/provider_yield.test.ts
test/worker_boundary.test.ts` initially produced two failures; the terminal
classifier's flat-envelope case subsequently failed before its compatibility
fix. `pytest -q tests/test_boundary_module_closure.py` failed until the staging
manifest included the classifier. Final focused coverage passed.

Native lifecycle qualification: FAILED, not PASSED. Four fresh attempts used
24.2864, 27.3357, 26.7516 and 21.3648 seconds. They released two, three, three
and one epochs respectively. In the third attempt both models completed all
three exchanges, the child requested yield, every GLM application matched both
rank digests and both native workers exited zero without forced termination;
OMP nevertheless reported a later context error. A controller exit code and
settled files alone are insufficient, so its independent verdict remains
FAILED. Other premature control closures remain unattributed; do not infer
that correcting the final-yield fixture resolves all those failures.

The context error was then reproduced without models through the installed
OMP executable, real provider and deterministic public-control worker events.
The executable advertises flat `data`/`error` yield arguments, but the adjacent
readable source advertises a `result` wrapper. The qualification extension
incorrectly rewrote a valid flat yield into that wrapper. OMP discarded its
fields, and the existing execution-receipt guard correctly rejected the
resulting history mismatch. Correcting the fixture to the executable's schema
removed the error: three ticks per actor, one successful yield, no blocked
tools, exit zero. This no-model replay does not qualify native exchange.
An attempted result-time receipt amendment did not fix the reproduction and
was removed; the production history-integrity checks were not weakened.

Final QA: `.venv/bin/python -m pytest -q` produced 1419 passed, three
environment skips and one existing warning in 86.16 s. Those skips remain
BLOCKED real-adapter gates. `bun test` produced 183 passed, 771 assertions,
zero failures in 3.42 s; `bunx tsc --noEmit` passed. `git diff --check` passed.
The final no-model replay used uninstrumented production files. Diagnostic
copies and raw native evidence remain in the private audit area, outside
publication scope and model-readable workspaces.

Evidence SHA-256: final Python log
`eda26709c054b6f900b90fa73848f0709ed9f3860576abf3c10974576ab40f54`;
plugin log `02c2684cc26053274df7ab275bf46c7827f2bfa6b39d57c7b0640c9a21a314e5`;
third-attempt independent verdict
`988f800d0e536e16b52fa8be5dd597ace469f14c54ec82fda5462f567816d2cd`;
installed executable
`cf8d34a7fe6f60de1acbe74f29c82026e4c07888e9d89f7ebceeb922159e5787`.

Budget reconciliation: the prior takeover attempts charged 6 + 26 + 30 s,
reducing the previously recorded 177 s to 115 s. This ticket conservatively
charges 25 + 28 + 27 + 22 = 102 s, leaving 13 s of the original 600 s.
The local deterministic replays loaded no model. No training ran; money and
energy were not measured. Bounded bridges/coordinators stopped, native workers
were reaped and owned control tunnels were closed; shared MCDMA daemons were
unchanged. No further full native attempt fits the remaining allowance.

GitHub skill verdict: DO NOT PUSH. Staged-change Gitleaks with decoding and
archive inspection plus contextual searches found no confirmed disclosure,
but this is not a full publication audit. The separate sanitized candidate
remains unchanged; the original development history must never be pushed.
Next ticket: obtain enough native allowance for the corrected, runtime-schema
pinned three-epoch proof and linked cancellation, investigate any remaining
premature closure, then audit the exact sanitized publication candidate.

## 2026-09-22 — Idempotent parent completion after native repetition

Stage: installed-OMP/native lifecycle repair, not a scientific milestone.
Implementation commit: `c08cf4c253648cf4c4327b2a481693f2b4ed92f3`.
Local repair: PASSED. Native lifecycle qualification: FAILED.

The user authorized another 60 seconds of native inference, added to the
remaining 13 seconds. Two attempts consumed 22.2017 and 31.5445 seconds;
conservatively charge 23 + 32 = 55 seconds, leaving 18 of the total 660.
The first stopped after one exchange with a premature control closure that
remains unattributed. The second completed all three MCDMA exchanges and both
final confirmations, then failed after OMP announced the child's completion
and generated a second parent no-tool response. The already-written terminal
marker was rejected as a duplicate. Private control-order instrumentation
identified that failure; no diagnostic logging was added to production files.

Both attempts independently reaped both workers with exit zero and without
forced termination. Every recorded GLM application matched both rank digests.
The second attempt had three ticks per actor and one successful child yield;
167 bytes of static task-control text were counted outside the activation
channel. This is controlled lifecycle evidence, not stock Duo qualification,
held-out recall, attribution or final-tail coverage. Linked cancellation was
not exercised during this ticket, and the repaired finalization path has not
yet been rerun natively. A complete repetition run does not fit the remaining
18 seconds. No training or shared service restart occurred; money and energy
were not measured.

The finalization gate now permits repeated completion only after both actors
have settled, revalidating all four pinned terminal records without another
confirmation or publication. New exchanges remain forbidden after settlement;
record tampering, cancellation and the original deadline still fail closed.
The regression failed before the repair and passes afterwards, including a
tampered peer-settled record. No timeout was extended.

Commands: `bun test test/provider_yield.test.ts` reproduced one failure;
`bun test test/provider_yield.test.ts test/project_boundary_epochs.test.ts`
passed six tests with 70 assertions. Final `bun test`: 183 passed, 775
assertions, zero failures in 2.23 s; `bunx tsc --noEmit` passed.
`.venv/bin/python -m pytest -q`: 1419 passed, three environment skips and one
existing warning in 61.73 s. The skips remain BLOCKED real-adapter gates.
`git diff --check` passed. Bounded bridges and coordinators expired, native
workers were independently reaped, and the owned API/Unix control tunnels
were closed. Both shared MCDMA targets and handoff daemons retained their PIDs.

Evidence SHA-256: first native verdict
`19aa7626fb66f2ffc10f4fff75c976647465774106d8a8bf83a0316471a6129b`;
traced native verdict
`c193a59187ddcaf0dcc628534e47c73e6cd32cfabcd680ce9433a49738f327ca`;
final Python log
`da45cc83fd0470e5fd64aa42ce2961beb41c917988d50be491098672189e8fcf`;
final plugin log
`6cacbae8b40788d6602707a5041b0ef416c2a2b0f3932b656d342b8f69e2d13a`.
Raw native evidence and runtime profiles remain in the private audit area,
outside publication scope and model-readable workspaces.

GitHub skill verdict: DO NOT PUSH. Staged-change secret scanning with recursive
decoding and contextual identifier review found no confirmed disclosure in
the repair; this does not constitute a full publication audit. The separate
sanitized candidate is unchanged, no content was sent to GitHub, and original
development history must not be pushed. Next ticket: within a newly authorized
native budget, rerun the corrected three-epoch lifecycle and linked
cancellation, diagnose any recurrent early closure, then audit the exact
sanitized candidate before publication.

## 2026-09-22 — Native three-epoch lifecycle and linked cancellation

Stage: bounded installed-OMP/native engineering qualification. PASSED for the
specified parent/subagent fixture, not a scientific-stage or general Duo pass.
Code revision: `d2638d8cb0c6967ee970d514f1c0e4498c4dedf4`, including repair
`c08cf4c253648cf4c4327b2a481693f2b4ed92f3`. Production copies were uninstrumented.

The native GLM parent and Qwen child completed three two-way MCDMA epochs,
three checkpoint tools each, one explicit yield and both final confirmations.
OMP exited zero without a forced stop; independent receipts verified both
native workers reaped with exit zero and every GLM application matching both
rank digests. Total wall time was 30.033970 s, including startup and generation.
The passing run emitted five parent terminal events, so the duplicate-finish
branch is proven by its local regression, not by this particular native run.

A separate installed-OMP RPC run released epoch zero, confirmed its application
at the next boundary and aborted with epoch one staged on both routes. Abort
was acknowledged; both routes failed closed; neither second checkpoint ran.
OMP and both native workers exited cleanly without forced termination, with
both-rank application receipts independently checked. Total wall time was
22.037742 s. These checks used 167 and 141 bytes respectively of fixed task
control arguments outside the activation channel; tool results and local
prompts also exist. This is not a whole-workflow no-text claim.

A preceding 0.920299 s startup attempt was INVALID: preparation pinned mailbox
sessions before the fresh bridge readers started, so the owner correctly
rejected stale session identities before inference. Fresh preparation after
bridge startup fixed that orchestration error; no session guard was relaxed.
The earlier unexplained control closure remains a repeated-run reliability
limit rather than being retroactively assigned this cause.

The additional 60-second authorization raised the remaining allowance from
18 to 78 seconds. Conservatively charge 1 + 31 + 23 = 55 seconds, leaving
23 of the aggregate 720 authorized seconds. No training or shared daemon
restart occurred, and no weights changed. Bounded bridges/coordinators stopped,
owned control tunnels closed and shared daemon PIDs stayed unchanged.
Money and energy were not measured.

Commands: focused `bun test test/provider_yield.test.ts
test/project_boundary_epochs.test.ts`, six passed and 70 assertions;
full `bun test`, 183 passed and 775 assertions in 2.18 s;
`bunx tsc --noEmit`, PASSED; `.venv/bin/python -m pytest -q`, 1419 passed,
three environment skips and one existing warning in 60.60 s.
The three skips remain BLOCKED real-adapter gates. `git diff --check` passed.
Private `run_local.py` and independent `collect.py` produced both native
verdicts; their configurations, process records and raw outputs remain outside
model-readable and publication directories.

Evidence SHA-256: three-epoch verdict
`5a989290ca0d249576070ca11ebb8a6b09591e8c6d82f1fbab5b3aa9e2935144`;
linked-cancellation verdict
`74387726d39af1d7c5be167e1f6f4434464731383f2989470201d43de6e8e329`;
Python suite log
`0462399eda44ea156dec3ee6fdd31906fcb39029aec53e9dffc7201448b250c1`;
plugin suite log
`f01f43158211bfad9a1d7812d726d25f92d97d0151666fe0d2a3eacf76dd5880`.

The README and native exchange guide now distinguish this bounded success
from automatic stock Duo startup, arbitrary nested tasks, held-out recall,
source attribution and general final-tail coverage, which remain unqualified.
Next ticket: audit and publish the sanitized source snapshot, retaining these
limits and excluding original development history and private receipts.
