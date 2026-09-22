# TP rank delivery guard

The coordinator now treats a head-rank write as insufficient evidence of delivery.
It hashes the outgoing publication and verifies the same file on every configured
host before renaming the head file that makes the publication schedulable.
After that rename it waits, within the configured I/O deadline, for each rank's
receipt containing the rank, TP world size, sequence, row count and file digest.
The receiver writes its receipt atomically only after every local layer write and
its existing CUDA synchronization have completed.

Any host error marker, unreachable probe, missing or changed publication,
mismatched receipt, or absent receipt at the deadline invalidates the run.
Periodic and final health checks inspect error-marker existence on every host;
they never read error-marker contents. No-link sessions do not perform these
probes or require receipts. Existing peer output directories are refused instead
of deleting their evidence.

The host order declares one TP rank per host: the head is rank 0, followed by the
configured peers. Receipts verify this declaration and the complete TP world
size; a different topology needs an explicit rank-to-host mapping before use.
The owner's inspected `_tp()` hook returns `(rank, world_size)`.

This is admission and failure detection, not an atomic TP transaction or a
scheduler barrier. A file may disappear after admission, or one rank may fail
after another has written; the session then becomes invalid, but writes are not
rolled back and speculative computation may already have occurred. Polling and
receipts do not prove source ownership, consistent cross-rank decode snapshots,
remote cancellation, or prior-epoch causality.

Receiver deployment must include `live_rank_receipt.py` alongside the flat GLM
receiver modules. No deployment or server restart was performed for this change;
a server lacking receipts fails the coordinator's acknowledgement deadline.

Synthetic tests cover peer errors, missing files, digest differences, missing or
incorrect receipts, actual rank identity, partial local writes, duplicate local
sequence rejection, and no-link behavior. The first four coordinator regressions
produced successful reports before the fix, then correctly invalidated the run.
Real TP execution and resource-release qualification remain blocked until an
owner-approved deployment and bounded live failure test are available.

Verification used base commit `c74ae70a03762d24041190e9158dac3090fff701`
in an isolated worktree, with the existing environments linked locally and
`PYTHONPATH` set to that worktree; both interpreters resolved `drift.__file__`
inside it. The focused command was
`PYTHONPATH="$PWD" .venv/bin/python -m pytest -q tests/test_live_rank_delivery.py tests/test_live_no_link.py tests/test_live_receivers.py tests/test_vllm_glm53_inject.py --tb=short`:
75 passed in 0.71 seconds. `PYTHONPATH="$PWD" ./scripts/check_all.sh` passed
with 269 reference tests and two existing skips in 19.20 seconds, 155 next-runtime
tests and the existing sparse-indexer-tie xfail in 7.91 seconds, and 29 plugin
tests in 66 milliseconds plus TypeScript checks. GPU/energy costs were not measured;
no remote mutation, serving restart, real-model inference or publication occurred.
The next gate is approved deployment and bounded real-TP failure qualification,
followed separately by a per-request causal scheduling barrier.

A follow-up on base `2303ef2e79b7bf1af7556a363bd4cf3dccc56d68` fixes the
acknowledgement deadline itself: each SSH probe receives only the time remaining
on one absolute deadline, and replies at or after expiry are rejected even when
all receipt fields match. Failure checks, parsing and pauses cannot reset this
budget. Four deterministic regressions failed before the fix; the focused suite
then passed 79 tests in 0.72 seconds. The same full-check command passed 273
reference tests with two skips in 19.01 seconds, 155 next-runtime tests with the
existing xfail in 7.68 seconds, and 29 plugin tests in 66 milliseconds plus
TypeScript checks. These remain local mechanical tests, without real inference.
