# Repair integration, 21 September 2026

This ticket integrates the local live-cache repair candidate with the newer exchange, cache-readback,
source-age positions, process leases and prefix-reuse work.
It does not deploy a connector, qualify a native runtime or establish reliable two-way recall.

The isolated integration started at `585a5581374b645ee57af17be16c20b014750efd`.
The concurrently committed prefix repair `5f829b12bd3f46bedeee4ffac4a78fccde568404`
and deployment pin `660b8b5ea33f596edd6a298f0628b35c4dfc92e2` were preserved.
The older candidate came from the `drift-live-cache-repairs` worktree based on
`172ceb6cbe2e11223adb5a0398d4b4e5a838cc22`; it was not copied over newer files wholesale.

## What changed

* Complete scheduler allocations replace old block lists; explicit preemption invalidates them.
  Live resumes refuse further writes and require fresh sessions, with the prefix repair's typed diagnostics retained.
* A new mailbox writer cannot report an ACK, adopt an occupied slot or silently reset an outstanding stream.
  The bridge keeps publication state through retryable transport errors and retries acknowledgements without repeating release.
* Cache-save failures participate in a tensor-parallel CPU-group failure vote and raise to the engine on every rank.
  Rank markers remain diagnostic evidence, not the only failure propagation mechanism.
* Sparse fan-out artifacts clamp to their supported contiguous residual prefix in NumPy and MLX.
* Terminal-range evidence and the existing accepted-cache-tail protocol remain intact.
  This does not claim that the last sampled but uncomputed token has a cache entry.
* Exchange snapshots are checked before delivery, including the actual body digest and cumulative outgoing capacity.
  Incoming source ranges are checked before mutation, and malformed receipt results poison the session.

The pre-existing foreign-row capacity fix in `585a558` is retained.
The early-ACK and accepted-tail fixes already absorbed into main were checked rather than replaced with older variants.
The existing source-age positioning and owned-process cleanup remain in place.

## Reproductions and verification

The pre-change main baseline passed 1161 tests with three environment skips in 58.28 seconds.
The imported repair regressions then reproduced 15 failures with three passes in 0.21 seconds,
including new-writer ACKs, stale ACK reuse, bridge termination on timeouts and swallowed rank failures.
The seven new exchange guard cases failed before their fixes in 0.03 seconds.
They drive the actual session methods and assert no publication, cache write or ACK after malformed input.

The first combined repair run passed 149 focused cases in 11.19 seconds, including a real local
two-process CPU Gloo failure vote with either rank failing.
The first full integration run exposed seven existing cache-commit tests that expected swallowed errors
and one sorted-list mismatch in the bundle test.
Those tests now require raised failure while retaining their no-write, no-receipt and poisoned-state assertions.
The final focused guard, prefix and cache-commit selection passed 59 tests in 0.12 seconds.
Final full-suite results, source hashes and exact commands are recorded in `docs/agent-progress.md`.

No debug instrumentation, backbone edits, host access, trained-checkpoint loading, deployment or service restart was used.
Test timings are measured wall times; money and energy are unmeasured.

## Remaining gates

The connector bundle manifest describes local source only.
Before deployment, inspect both hosts' ownership, active source hashes, vLLM CPU-group API and rollback inventory,
then obtain the coordinated rollout and bounded native qualification window.
Local Gloo checks do not prove the deployed multi-host collective works or measure its per-step overhead.

The next local ticket binds the real transport/runtime return shapes to the exchange contract,
then implements the persistent coordinator and its bounded Duo boundary client.
The current classes are not drop-in compatible, and no adapter or coordinator completion is claimed here.
Reliable two-way recall, native cache-consumer selection and native OMP/Pi subagent integration remain BLOCKED.
This repair is not publication clearance.
