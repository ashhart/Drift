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
and never release buffered tools. This remains a one-shot project boundary, not an automatic multi-epoch
controller; subsequent boundaries require a separately qualified scheduling design.

## Not yet established

Composed tests now use the real mailbox decoder and live adapters with synthetic cache writes, the actual JS
client against the Python coordinator, and the production registration factory with the real parking gate
and a synthetic socket peer. They preserve failure, capacity, completion and cancellation checks.
The repaired connector bundle has since passed both-rank deployment checks and a native no-link
cancellation run, as recorded in the progress log. That does not qualify a native MLX/MCDMA exchange
or a real Duo turn. The remaining integration needs a runtime-owned coordinator bootstrap, verified
private transport configuration and a bounded two-actor proof with independent cleanup receipts.
The old linked-round script transfers activations over SSH and cannot qualify the MCDMA path.
