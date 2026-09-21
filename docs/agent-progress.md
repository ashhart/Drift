# Agent progress

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
