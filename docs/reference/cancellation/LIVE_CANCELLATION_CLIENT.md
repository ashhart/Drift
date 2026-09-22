# Aggregate-only owned-session cancellation client

This is a local qualification building block, not a model launcher that runs on
import and not a server-memory-release verdict. Its first gate is one owned
`--no-link` request using the real Spark runner's `--control-stdin` path; activation
imports/exports remain disabled. The parent orchestrator supplies the approved
command, on-host credentials, staging hashes, metrics samples and resource window.
No server, connector, checkpoint or deployed module is changed by this client.

## API

`drift.serving.live_qualification_client` exports `Limits`, `claim_session`,
`OwnedSession` and `QualificationError`; event accounting and session claims live
in focused companion modules. `claim_session(private_ledger_root)` creates a fresh
UUID-named persistent reservation, and `start()` atomically spends it before any
process launch. Reservation markers are never deleted or retried automatically.

Construct `OwnedSession(command, claim, limits)`, then call `start()`, repeatedly
`poll(timeout=...)` (or `pump`), inspect `request_state()`, explicitly `abort()` only
after the external observer qualifies activity, and always `close()` in cleanup.
`report()` returns only fixed labels, counts, byte totals, elapsed time, local child
PID/exit/termination evidence and requested token limits. No generated text, raw
events, raw stderr or private input is stored in a report or exception message.

The command must contain exactly one explicit `--session` matching the claim,
`--max-new` matching `Limits.max_new`, `--reserve` within the prompt budget,
`--no-link`, and `--control-stdin`; ambiguous equals-form overrides are rejected.
The actual runner remains responsible for putting the requested generation limit
in the server request. Output chunks are SSE events, not tokens: observed generated
token count is deliberately null, and the report names the requested cap separately.
Prompt counts include the reserved placeholder budget when enforcing the limit.

## Evidence boundary

The duck-typed state has `started`, `active`, `abort_sent`, `cancelled`, `completed`
and `concurrent`. `started` means the local supervisor was launched, preventing
duplicate admission while tokenization is slow; the runner's earlier `started`
event is counted separately and proves tokenization only. `active` requires nonempty
output, a live local supervisor and no terminal event, abort or protocol failure.
External metrics must also establish isolated server activity before abort; this
client alone cannot detect other server clients or prove request ownership in a
shared metric counter. Its `concurrent` field is false because it admits only one
local child, not because server-wide exclusivity has been established.

Production `spark_live_session.py` now distinguishes outcomes after its existing
supervisor cleanup: SessionCancelled emits only `{"cancelled": true}` and exits 2;
SessionControlError emits only `{"failed": true}` and exits 3. Natural success still
emits done and exits 0, and the unsupervised default is unchanged. The client sets
cancelled only after its own abort, that typed acknowledgement, exit 2 and stdout
EOF, with no done event or client failure. Exit 2 alone is not cancellation, and
unexpected exit/EOF without the proper terminal event fails promptly.

Natural completion is retained as completed and cannot pass cancellation. A narrow
race between the last observed activity and the abort still requires the external
observer and the actual server's metrics/release evidence; local child termination
is not an API-level server cancellation receipt or proof of GPU memory release.

## Bounds and cleanup

`Limits` bounds requested new tokens, prompt tokens, event count, line bytes,
aggregate output bytes, monotonic elapsed time and stop waits. Polling reserves a
cleanup allowance from the wall budget (normally 2 × stop_timeout + 0.25 seconds);
protocol or limit failures terminate/reap the owned local child before raising a
fixed error code. `close()` escalates to kill only after its bounded grace, and
reports whether escalation was used. The external orchestrator must keep pumping
and bound its own metrics calls; an idle caller is not an autonomous watchdog.
Observed elapsed time includes cold start and cleanup, with no silent warmup or
whole-request retry. A failed or over-budget cleanup is invalid evidence, never
converted into a success by dropping its time or output.

Synthetic tests use a local HTTP/SSE server and fake scripts only. They cover typed
abort versus invalid control, delayed tokenizer start, empty output, absent/unsolicited
acknowledgements, natural completion, bounded line/events/output/wall behavior,
local process reaping and one-use claims. Real cancellation and server-memory
release remain BLOCKED pending the independently controlled live qualification.
