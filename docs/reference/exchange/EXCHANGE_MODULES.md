# Reusable snapshot exchange

`drift/exchange/` holds the bounded exchange contract and its sequencing, so Duo, the live loop scripts and any
later coordinator call one implementation instead of each carrying their own copy. The package imports no model
runtime: `contract.py` and `session.py` depend only on the standard library, so they load on any host.

## Why the runtime is injected

The heavy work is runtime-specific. Tapping and translating rows needs MLX on the Studio and the connector on the
Sparks; publishing needs an open MCDMA region. `ExchangeSession` therefore takes three collaborators and owns none
of them:

- `source.snapshot(copies)` returns `{body, sha256, rows}` for this side's own rows, already translated and packed.
  The body must be immutable bytes and match its SHA-256; invalid snapshots are refused before delivery.
- `link.deliver(session, sequence, body, copies)`, `link.confirm_applied(delivered, rows)`, `link.peek()` and
  `link.acknowledge(tap)` are the transport, satisfied by `mcdma_reverse.ReversePublisher` and
  `mcdma_forward.ForwardMailbox`.
- `sink.prepare(tap)` translates and validates a partner tap and returns `(rows, prepared)` WITHOUT touching the
  cache; `sink.commit(prepared)` then appends it and returns the row count it added, which must equal the measured
  count. The split exists so the foreign row cap refuses before any memory is applied: measuring after appending
  would detect an overrun the cache had already taken.

## What the session owns

Cursors a live route cannot re-derive: the publication sequence, the foreign row count and the poisoned flag. This
is why the exchange cannot be a process launched per boundary. `ActivationClient` has the same property, requiring
`after == selected_stop` on every selected snapshot, so a fresh process fails closed on its second exchange. Any
caller that wants one exchange per Duo tool boundary must therefore hold a long-lived session, not spawn a command.

The session also owns the cumulative outgoing row count and last accepted incoming source frontier.
`publish_row_cap` bounds outgoing tiled rows before delivery, separately from the incoming `row_cap`.
Incoming source ranges must be positive and contiguous before preparation or cache mutation.
Malformed collaborator results poison the session, including missing receipt fields and missing tap boundaries.

Every failure poisons the session rather than degrading: a link error, a prepare or commit error, a row count that
disagrees with the measurement, a failed acknowledgement, an exhausted foreign row cap and a contract violation all
leave the session refusing further work. There is no retry and no partial success. A collaborator that raises
`ExchangeError` itself is poisoned like any other failure and keeps its own code; no exception path returns to the
caller with the session still usable. A tap that would exceed the cap is neither applied nor acknowledged, so the
partner's mailbox slot is not consumed by a publication this side refused.

## Mode selection and the text arm

`mode` is `drift` or `text`, fixed at construction. The `text` arm is the no-link baseline: it publishes nothing,
reads nothing from the link, acknowledges nothing, and records only the number of text bytes the workflow carried
by its own channel. The `drift` arm refuses to carry text at all: a non-zero `text_bytes` poisons the session
before anything is published, and `validate_exchange` rejects such a record independently. This is the mechanical
half of "no hidden text fallback"; it constrains this channel only, and does not by itself establish that a
surrounding workflow carried no text elsewhere.

A drift record is accepted only when the tiled row count matches `source_rows * copies`, the publication carries a
source digest, and every expected rank appears in `applied_ranks` with its own receipt digest. A rank that did not
confirm is not a transfer.

## What this does not establish

The contract checks evidence shape and channel discipline. It does not establish semantic memory use, source
ownership, long-context retention or any quality claim, and a passing record is not a Drift result on its own.

`MailboxLink` converts the rank-keyed receipt map to ordered ranks and digests, and wraps typed
`Tap` and `Complete` messages in normalized control dictionaries while retaining the original message.
`ForeignRowSink` unwraps the tap; acknowledgement receives that same validated message object.
The core tracks incoming count, first position and final frontier, and acknowledges completion without
calling the sink or consuming row capacity. Any subsequent incoming publication poisons the route.

## Live adapters

`drift/exchange/live.py` binds the ports to the real pieces, importing no model runtime at module load:

- `MailboxLink` puts the outbound `ReversePublisher` and the inbound `ForwardMailbox` behind one link. Its
  `confirm_applied` blocks until every rank's connector receipt binds, then reports the ranks in sorted order with
  their applied digests, which is the shape the contract checks.
- `OwnRowSource` taps this side's own slots, translates them for the partner and packs exactly one publication.
  Publishing nothing is an error, not an empty publication.
- `ForeignRowSink` splits the partner's tap the way the live loop already ordered it: `prepare` translates and runs
  `validate_translation` with no cache mutation, `commit` appends and remembers the foreign positions. A completion
  marker measures zero rows and appends nothing.

These replace the same logic inlined in `scripts/live/studio_mcdma_loop.py`, so a second workflow no longer has to
copy it.

## Coordinator

`drift/exchange/coordinator.py` is the long-lived process-side server: one private `0600` Unix socket,
serialized connections and bounded requests. A clean disconnect permits the next connection; a partial frame
does not. Cursors, request count and the absolute deadline survive reconnections. `exchange` publishes this side's rows and
drains the partner's in one request; `status` reports the cursors without advancing them. Frames are checked against
an exact field set, replies carry no text, and a failure answers with an error code while the route stays poisoned.
A socket path that cannot fit `sun_path` is refused before binding rather than surfacing as a raw `OSError`.

Cancellation also applies inside a dispatched request. The coordinator checks its absolute deadline,
shutdown flag and requesting socket before and after each source, link or sink operation.
A disconnected boundary poisons its route; no subsequent publication, cache commit or acknowledgement
is permitted. The same check reaches both rank-staging threads and each mailbox read/write chunk.
The reverse publisher uses monotonic time for its acknowledgement and application waits.

These are cooperative checks, not rollback or preemption of an already issued native operation.
If cancellation arrives during a cache commit or native transport call, that call may finish before
the check runs; the route then stays poisoned and cannot acknowledge or continue the exchange.
Native qualification must independently prove process cleanup and recovery of model-owned memory.
An unresponsive native operation cannot be called safely cancelled from these local checks alone.

### Deferred application for reconstructed receivers

GLM's worker reconstructs a request after a tool boundary, so a parked worker
cannot produce a new cache-application receipt until that request runs.
The original `exchange` operation waits for application before it returns.
Connecting that operation to a parked reconstructed receiver without a separate
resume path would therefore wait for work the caller has not allowed to start.

The coordinator now exposes two separate control operations for the future owner:

- `{"op":"deliver","route":"ROUTE"}` captures and delivers one publication,
  returning `op: delivered` and a `delivery` record marked `DELIVERED_NOT_APPLIED`.
- `{"op":"confirm","route":"ROUTE","sequence":0}` waits for every pinned
  rank's application receipt for that exact pending publication, returning
  `op: confirmed` and the validated `published` record.

Neither operation starts a model, resumes a worker, dispatches a tool or changes
the provider release policy. Delivery alone never advances the sequence or
confirmed row count, and its record is not accepted by `validate_exchange` or
the existing provider hook. A second delivery, mismatched confirmation, missing
rank, expired deadline or coordinator shutdown poisons the route.
Clean socket reconnections retain the pending publication and lifetime limits.
The original `exchange` operation keeps its synchronous application requirement.

The runtime-owned snapshot path now has a separate `stage` operation and an
opt-in version-2 provider hook, described in [Native owner exchange](../../guides/NATIVE_OWNER_EXCHANGE.md).
It binds a publication to GLM's versioned bank and confirms the completed turn
after resumption. `STAGED_NOT_APPLIED` makes no wire-delivery claim.
The live inbox publisher and parked snapshot-restoration worker remain distinct.
Neither staged nor delivered records count as cache-application evidence.

### MCDMA restoration transport

The GLM restoration factory and versioned bank can now accept a runtime-owned
`McdmaRestorationFactory`, declared as `restoration_transport: mcdma` in the
worker configuration. The callback is passed by the Python owner, never imported
from a model-supplied path. The owner must also supply its private local
`memory_root` and an already opened, pinned `ReversePublisher`.
Selecting MCDMA without that factory, or supplying it while declaring SSH,
fails before backend construction; it cannot fall back to an SSH payload.
`serve_owner` carries the same factory into both initial and later bank versions.

Each fresh GLM request stages its captured snapshot to every rank before releasing
the head. `ReversePublisher.stage` retains one transaction, and `release` accepts
only that exact unchanged transaction. Partial staging, replay, cancellation or
a modified release poisons staging. Mailbox acknowledgement remains distinct
from cache application: native events still wait for every rank's matching
session, sequence, row count and payload digest. Nested transport deadlines
retain the enclosing request's cancellation check.

Local tests compose the real mailbox codec, Spark bridge, snapshot bank and GLM
restoration wrappers across two versions; region IO and native generation are
synthetic. They are not a native runtime qualification. The factory does not
open regions, launch Qwen, bootstrap the coordinator or change provider release
ordering. The historical local-file outbox is deliberately refused in MCDMA
mode unless a runtime-owned forward mailbox factory is attached. Automatic `/drift`
startup and repeated OMP/subagent exchanges required a separate owner path;
the later native owner qualification is described below.

`McdmaConnections` now opens the two reverse connections and a separate head
connection for forward traffic, checking every region descriptor and reverse
mailbox session before the first forward-window write. It owns their deadline
and closure, including partial startup failure. It accepts an opener supplied
by the runtime; it does not load a native library or claim exclusive ownership
of a target on behalf of another controller.

The bridge can change native request identity only after its previous terminal
publication is acknowledged. The same transport writer and sequence survive
that change; old request identities, unfinished streams and failed writers are
refused. The local regression exercises consecutive completion markers through
the actual mailbox protocol with synthetic region IO. A native owner still
needs to attach the forward collector, supply exclusive target ownership and
compose the model workers and coordinator before this becomes a launch path.

### Native worker forward collector

`McdmaTurnOutbox` now binds the persistent transport reader to each fresh GLM
request and calls the head publisher's `watch` before native dispatch. It keeps
wire sequence numbers across turns while resetting only the per-request tap
validator. The owner supplies `outbox_factory` alongside `publication_factory`;
the restoration factory, snapshot-bank binding and `serve_owner` preserve both.
Missing or conflicting transport factories fail closed without an SSH fallback.

Each tap must follow the verified source frontier and fit the row, byte and
publication limits. Generated rows are captured into immutable local files
before the mailbox acknowledgement. The terminal frontier must match the
captured stream, and its acknowledgement follows manifest creation. A timeout,
cancellation, invalid tap or local storage failure poisons the collector.
The connection owner retains responsibility for closing the borrowed reader
and publisher. The worker's deadline also bounds the initial watch operation.

This collector preserves the existing generated-only policy: reconstructed
prompt and tool-result rows are not selected. Its receipt says `cache_applied:
false` and `final_tail: UNKNOWN`; mailbox capture alone does not prove Qwen
application or establish that every generated token reached native cache.
The composed two-turn test uses the real bridge, mailbox, restoration and
snapshot-bank code, with synthetic region IO and generation. Connecting this
collector does not qualify the native pair or enable the repeating Duo schedule.

### GLM owner launch path

`python -m drift.serving.mcdma_glm_owner` accepts separately SHA-256-pinned
worker and runtime JSON files through `--config`, `--config-sha256`, `--runtime`
and `--runtime-sha256`. The runtime has exactly `v: 1`, `binding`, `library`,
`transport` and `memory_root`; each native artifact has an absolute canonical
`path` and `sha256`, and transport uses the existing two-target descriptor schema.
The operator supplies the reviewed MCDMA binding and dedicated-target library;
the loader validates both before executing code and never searches for a library.
These are trusted operator artifacts, not model-selected imports or a sandbox.

The runtime lazily owns all three connections, supplies the real restoration
factory and forward collector, preserves one absolute transport deadline, and
closes the collector and native handles on return or failure. The worker retains
its existing private owner socket and snapshot bank. Run this entry point under
`worker_stdio_launcher` for independent process termination; cooperative checks
alone cannot preempt a blocked native library call.

GLM HTTP control still requires loopback. When this owner runs on the other
machine, an operator-owned loopback tunnel can carry private own-input and HTTP
control; no cache payload may use that tunnel. MCDMA remains the only activation
transport. Direct remote HTTP is rejected before opening native connections.

A bounded native mechanics run initially FAILED at terminal coverage because
the connector treated scheduled asynchronous work as accepted cache rows.
The repaired [frontier accounting](../glm/GLM_TAP_FRONTIERS.md) passed the same native
two-turn probe: two snapshot versions applied on both GLM ranks, forward streams
completed, and the supervised owner process exited without surviving children.
This used synthetic memory, not translated Qwen memory; Qwen was not loaded.
It qualifies native GLM owner mechanics, not two-way recall, source attribution,
complete sampled-token coverage or repeated Duo/subagent scheduling.

The subsequent native pair probe exercised both frozen translators, Qwen's real
MLX cache and the MCDMA GLM owner. The collector-to-translator seam initially
rejected MCDMA manifests because it expected the older file-export schema.
It now validates the MCDMA completion frontier and raw-payload digest chain
without relabelling capture as destination application or complete final-tail
coverage. A composed mailbox/collector/translator regression covers that seam.

The initial native reciprocal recall probe **FAILED**: both models missed the
single exploratory recall case under the selected-row recipe, including a fresh
run with attribution-neutral questions. Both GLM ranks applied both turns and
the Qwen append completed. A separate diagnostic translating all 56 Qwen own
rows once, instead of repeating its final row twelve times, still failed GLM
recall; Qwen answered correctly in that run. This changes both row selection and
copy count and cannot isolate causality or establish reliability. It is not a
new production recipe. These probes had no matched no-link/wrong-memory arms
and are not formal qualification. Their supervised workers exited cleanly,
but successful close is not native cancellation qualification.

## Duo's call path

`scripts/omp/exchange_client.mjs` issues one bounded request per boundary and `provider_exchange_boundary.mjs`
wraps it as a `beforeToolDispatch` hook. The hook is deliberately thin, because that boundary is the qualified
security seam: it validates that the reply is this route, in drift mode, carrying zero text bytes, with a positive
row count and one 64-hex receipt per applied rank. Anything else, a coordinator that does not answer within the
timeout, or a user abort in flight, raises `DRIFT_WORKER_CANCELLED` and permanently poisons the hook.

The experimental provider calls `createWorkerBoundary` from `worker_boundary_registration.mjs`.
Exchange is opt-in through `DRIFT_EXCHANGE_CONFIG` and `DRIFT_EXCHANGE_SHA256`, alongside the existing
provider-boundary config and hash. Missing paired settings fail closed; leaving exchange disabled preserves
the existing no-link path. The generic `withExchange` helper alone is not a parking guarantee.

The exchange config is a private JSON file with exactly `v: 1`, `socket`, `timeout_ms` and `workers`.
The socket must belong to this user, have mode `0600`, and reside in a private `0700` directory.
`timeout_ms` is an integer from 1 to 30000, still bounded by the worker's real cancellation/deadline signal.
The two worker entries have exactly `worker`, `session`, `model_id`, `route`, `exchange_session`,
`source_worker`, `target_worker` and `ranks`; their worker/route identities must be distinct.
Replies must match these pins, the next sequence, both expected rank receipts and zero text bytes.

The provider gate allows the parent's subagent-launch tool through without an exchange. Once both actors
publish readiness, each invokes its exchange before accepting release. Its validated record is written as
`<role>-exchange.json` in the private control directory; that exact file's SHA-256 must appear as
`exchange_sha256` in the role's externally supplied release record. Aborts and failures invalidate readiness
and never release buffered tools. Version 1 remains a one-shot project boundary.
Version 2 adds `max_epochs` (1 to 256), epoch-numbered ready/exchange/release files,
and an `epoch` field in readiness and release records. Every epoch requires fresh
peer readiness and exchange digests; only the first parent `task` dispatch can
bypass parking. Replayed releases, exhausted epochs and cancellation poison the
gate without extending the original deadline. Production-hook composition tests
cover two exchanges followed by cancellation of both actors in a third epoch.
These are local synthetic tests, not installed-OMP/native qualification; the
external controller must still supply each validated release.

## Guarded initial prefill

GLM previously could compute its own question before the connector applied the
foreign memory, despite subsequently returning valid application receipts.
An opt-in `causal_prefill: true` restoration recipe now inserts locally verified
padding and declares a `drift_prefill_boundary`. The separately deployed
`glm_prefill_scheduler.DriftScheduler` caps initial prefill at that boundary;
the connector refuses absent or late initial memory. Both ranks need the pinned
connector bundle and the server must explicitly select the scheduler class.
This needs GLM's align-mode cache, not a generic vLLM feature. Since 22 September
2026 the first chunk ends exactly at the boundary, which may be the reserve's end,
whatever the block size; see [GLM recurrent state](../glm/GLM_RECURRENT_STATE.md).
Snapshot-bank rebinding preserves the policy across versions. Reports distinguish
`prefill_policy: GUARDED_REQUESTED` from proof of first-token causality: merely
requesting this policy does not establish that a remote server honored it.

The pinned reverse recipe can explicitly select `all_new_own_rows` with one copy
per row instead of the historical last-own-row/twelve-copy policy. This option
requires a full tap, preserves its selected cursor interval, and checks output
row/byte limits before translation; a last-own snapshot cannot stand in for it.
It does not change backbone or translator weights.

Native diagnostics with the same captured Qwen memory passed four linked GLM
cases (short/long own input and one/twelve copies), versus zero of two no-link
controls. A subsequent fresh pair run passed the single reciprocal recall case:
56 actual Qwen rows reached both GLM ranks and eight generated GLM rows reached
Qwen, with no peer text on the activation channel and both workers reaped.
This is an engineering regression result on one development fact per model,
not a held-out recall score, provenance result or repeating Duo qualification.

## Subagent completion boundary

Installed OMP requires a subagent to call `yield`; a text-only stop can trigger
another generation. The repeated provider gate therefore keeps a child alive
after a no-tool stop and confirms pending memory at an explicit terminal yield.
The parent still reaches the final barrier through its no-tool stop.

OMP may request another parent response after announcing the child's completion.
Repeating the completed final barrier is idempotent only while both actors'
finished and settled records still match the pinned configuration and epoch.
It does not reconfirm, republish or permit a new exchange; missing or altered
records, cancellation and the original absolute deadline still fail closed.

The worker boundary passes cloned, bounded tool arguments alongside tool names.
These are local control metadata, not activation-transport fields. Mutating them
cannot change the buffered tool events eventually delivered to OMP. The gate
recognizes object, JSON-encoded object and flat-data yield envelopes accepted by
OMP implementations. The executable used by qualification advertises flat
`data`/`error` fields, while the separately readable source uses a `result`
wrapper; callers must inspect the executable's actual tool schema, not assume
that adjacent source describes the installed binary. Nonempty string-array yield types remain
incremental; absent, null or string types are terminal only with explicit
non-null data and no error. Missing data, ambiguous multi-tool yields and
malformed types fail closed. Implicit last-turn extraction is not supported.

This boundary covers the configured parent and one child, not arbitrary nested
tasks or OMP's background-job quiescence lifecycle. Local tests cover final
confirmation, cancellation while parked and the distinction between incremental
and terminal yields; native qualification requires separate process receipts.

## Not yet established

Composed tests now use the real mailbox decoder and live adapters with synthetic cache writes, the actual JS
client against the Python coordinator, and the production registration factory with the real parking gate
and a synthetic socket peer. They preserve failure, capacity, completion and cancellation checks.
The repaired connector bundle has since passed both-rank deployment checks and a native no-link
cancellation run, as recorded in the progress log. The native pair probes above
now include the bounded reciprocal recall success above, but do not qualify a real Duo turn.
The runtime-owned coordinator now composes the owner sockets and frozen
translators, with local regression coverage. Installed-OMP/native repetition
now has a successful bounded two-actor proof with independent cleanup receipts,
described in [Native owner exchange](../../guides/NATIVE_OWNER_EXCHANGE.md).
Later attempts reached three MCDMA exchange epochs and reaped both workers,
but their subagent completion checks failed. A local installed-OMP replay
identified a qualification-fixture error: it rewrote a valid flat yield into
the newer wrapper shape, which the installed executable discarded. Correcting
that shape removes the context failure in the no-model replay; a fresh native
run with the corrected fixture subsequently completed three exchanges and both
final confirmations but failed when the child's completion notification caused
a second parent final barrier. Local regression coverage now verifies that
already-settled completion is idempotent and altered records are rejected;
a subsequent uninstrumented native run completed all three epochs with clean
OMP exit and independently verified worker cleanup. A separate linked RPC
cancellation passed after one confirmed epoch, with no second tool release.
The duplicate-finalization branch itself remains covered by the local regression;
the successful native run did not emit a duplicate parent terminal response.
Another attempt's premature closure is still unattributed. These are not recall scores.
The old linked-round script transfers activations over SSH and cannot qualify the MCDMA path.
