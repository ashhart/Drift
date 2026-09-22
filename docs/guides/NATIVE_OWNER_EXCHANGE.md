# Native owner exchange

Engineering status: local mechanics PASSED; bounded native OMP repetition and
linked cancellation PASSED on the tested GLM/Qwen stack. This does not enable
an automatic `/drift` workflow or qualify arbitrary tasks and model pairs.

## Staging and application

The old synchronous hook waited for GLM to apply memory while holding GLM at
its tool boundary. GLM reconstructs its next request only after that boundary
releases, so the hook waited for work it prevented from starting.

`NativeOwnerCoordinator` composes two persistent owner-socket routes without
loading models or restarting services. Its trusted private profile pins each
owner's socket, memory directory, session and worker identity, plus the frozen
translation recipe and artifacts. The operator owns worker startup, exclusive
transport access, absolute deadline and cleanup. A profile is not model input.

The Qwen-to-GLM route selects Qwen's last own row, applies the historical
twelve-copy reverse recipe and publishes a new GLM snapshot-bank version.
The GLM-to-Qwen route translates the latest unseen completed MCDMA collector
manifest and appends the translated rows through Qwen's activation owner.
Neither route sends activation bytes through SSH. Files belong to the local
owner process; GLM's existing MCDMA restoration transports its bank snapshot.

`stage` returns `STAGED_NOT_APPLIED`. This means only that the owner accepted
the publication, not that bytes crossed the wire or either model used them.
The next boundary first confirms the preceding publication, then stages a new
one. GLM confirmation reads only completed-turn receipts and requires both
ranks to match the pending version, digest and row count. Qwen confirmation
requires its exact append receipt, including session, worker IDs and sequence.
No confirmation means no sequence or confirmed-row advance.

The source-row count in these exchange records describes translated publication
rows. Separate source reports retain the selected own-row interval and recipe.
This last-row policy does not establish full memory coverage or recall quality;
the earlier full-own-row reciprocal diagnostic remains separate evidence.

## Provider lifecycle

Exchange configuration version 1 keeps synchronous behavior. Version 2 requires
`delivery: next_turn_snapshot` in both worker entries and uses stage/confirm.
All existing socket ownership, configuration hash, worker identity, cancellation
and deadline checks still apply. Staged records cannot pass the applied-exchange
validator. Unexpected fields, receipt mismatch, concurrent calls or abort poison
the hook; it never retries the same publication as a fresh session.

The repeated provider gate waits for both actors to finish generation, confirms
their final pending publications and waits for both settlement records before
exposing terminal completion. This prevents parent teardown from closing a
peer that is still applying its final epoch. It does not grant more time.
A worker startup failure now poisons its project boundary even if it failed
before the first dispatch hook, so the peer cannot wait silently until timeout.

## Evidence and limits

Local regressions compose a real GLM snapshot bank, owner socket dispatch,
exchange session and staged confirmation. A subprocess owner test checks
PENDING before generation and exact both-rank receipts after terminal.
Production-hook tests cover repeated stage/confirm, final settlement, startup
failure propagation, bad receipts, cancellation and refusal after poisoning.
Native cache IO and model generation in these regression tests are synthetic.

An earlier installed-OMP attempt used a constrained task subagent and actual native
worker commands. The GLM parent launched its subagent, but the Qwen worker
failed before emitting output. No exchange epoch released; the parent reached
its deadline. The supervisor reaped it, but its orderly owner report was absent,
so neither repetition nor graceful native cleanup passed. The startup-failure
propagation fix followed this run and has only local regression evidence.

The next preflight found insufficient reclaimable Studio memory for the
unchanged model-plus-reserve guard while another workload was resident.
The guard was not weakened and the other workload was not stopped.
Those failures are historical, not the current bounded lifecycle verdict.

On 2026-09-22, the installed OMP executable completed three two-way MCDMA
epochs with a native GLM parent and Qwen subagent, three checkpoint tools per
actor and one explicit child yield. Final confirmations completed, every GLM
application matched both rank digests, OMP exited zero and both native workers
were independently reaped without forced termination. Wall time was 30.034 s,
including startup and generation, not per-epoch transport latency.

A separate RPC cancellation run released one epoch, confirmed its application
at the next boundary, then aborted while both actors had staged the second
epoch. OMP acknowledged the abort, neither second checkpoint executed, both
routes failed closed and both native workers were reaped without forced
termination. Total wall time was 22.038 s. No shared daemon was restarted.

These checks use constrained lifecycle instructions and tool results, with
167 and 141 bytes respectively of static task-control arguments counted
outside the activation channel. They do not establish text-free communication
across the entire workflow. Raw receipts remain private; aggregate evidence
hashes and the code revision are recorded in the progress log.

Matched recall controls, source attribution, general final-tail coverage and
automatic stock Duo startup remain separate gates. The earlier single-fact
reciprocal recall check is not a held-out reliability score. One preceding
run's early control closure remains unattributed, so this successful bounded
run is not a claim of repeated-run reliability.
