# Studio activation control and owned-process supervision

This is local implementation evidence for separate activation delivery and bounded
worker ownership, not a real GLM/Qwen collaboration result. It extends the concrete
Studio worker mode without adding activation fields to the OMP own-input protocol.
The private own protocol's first command sequence is now one, matching the actual
TypeScript provider; the mismatch was reproduced against the real Python core.

## Separate inherited activation channel

The owner configuration may supply `activation` with `fd`, `session`,
`source_worker`, `target_worker`, `memory_root`, `max_bytes`, `max_rows` and
`max_total_rows`. The FD must be an inherited connected AF_UNIX stream socket,
separate from own-input/output and supervision control/evidence pipes. It is an
OS capability, not a public listener; no network endpoint or prompt authenticates
it. The target must equal the configured own worker, and that worker's session is
bound to the activation session. The controller waits for `opened` before append.
The immutable route labels accompany receipts outside token input: append uses
the configured incoming source→target route, while tap reverses it to identify
the receiver's own-cache export. The client rejects swapped or forged routes;
these labels do not establish that either model understands source ownership.

The private memory root must already exist, have canonical non-symlink identity,
be owned by the worker UID, and deny group/other access. Its model-tool/evaluator
isolation still requires the deployment's distinct OS identities and permissions.
Only single relative `.npz` basenames are accepted; traversal, symlinks, FIFOs,
wrong sessions, duplicates/out-of-order command sequences, extra fields and bad
layouts poison the native session. Activation command sequence starts at one.

Append frame fields are `v:1`, `session`, `seq`, `op:"append"`, `path`, `sha256`
and `rows`. Input is copied into a fresh private snapshot, bounded and hashed,
then validated across every declared K/V layer before the existing append handler
runs. File and expanded archive bytes, per-publication rows and total appended
rows are bounded. The receipt has `op:"appended"`, actual rows, verified digest,
foreign total and bound identity metadata; it contains no tensors or task text.

Tap frame fields are `v:1`, `session`, `seq`, `op:"tap"`, `path`, `first` and
`max_rows`. The first cursor must match the previous tap's next cursor. A native
transaction checks the exact current own-row count and conservative output-byte
bound before writing. Output is validated and atomically linked to an exclusive
new filename, with no overwrite. The receipt includes rows, next cursor and hash.
The maximum is a refusal bound, not silent truncation; the controller must keep up
or end the session if the accumulated own rows exceed its bound.

## Serialization and poison

All production Studio handle calls share `NativeGate`: opening, own prefill,
one-token generation, continuation, append and tap. The lock is held through the
real `mx.eval` callback; activation mutations therefore occur between generation
steps, and receipts follow device completion. The gate's monotonic poison flag
cannot be cleared by an inner receiver resetting its own temporary state flag.
Waiting for a busy native gate is bounded to two seconds; a slow/stuck kernel or
expired wait invalidates this session rather than admitting concurrent writes.
No receipt implies recovery from a lost reply; use a fresh session after failure.

An activation EOF, malformed line, receiver exception or bounded reply failure
poisons the gate and cancels the backend. Queued commands cannot clear that poison.
The server bounds lines to 4 KiB and owns its duplicated activation FD. Backend
close serializes cache destruction through the same gate. The independent process
supervisor is still necessary to stop a blocked kernel or an idle own-input reader;
thread cancellation alone is not process or GPU-memory-release evidence.

## On-host supervisor

`python -m drift.serving.worker_supervisor --control-fd N --evidence-fd N
--wall-seconds SECONDS [--input-fd N --output-fd N --activation-fd N]
-- command argv...` launches exactly the requested child in a new process group.
The matching Python API is `supervise_worker(command, *, control_fd, evidence_fd,
wall_seconds, input_fd=None, output_fd=None, activation_fd=None, stop_timeout=2,
max_pending_bytes=262144, max_total_bytes=16777216)`.

Without input/output FDs, a one-shot child gets DEVNULL input and its stdout is
discarded with byte accounting. With them, the supervisor relays private bytes
through bounded queues, without parsing, recording or returning their content.
Only the activation FD is explicitly inherited by the child; ownership transfers
and the supervisor closes its copy after spawn. The controller must be the only
holder of the supervision-control writer so its death produces EOF on-host.
The fixture uses this exact executable route with distinct pipes and socketpair.

Control accepts only `{"op":"abort"}` followed by newline. Control EOF, own-input
EOF, output loss, signals, protocol/byte limits or the wall timer initiate cleanup.
The supervisor reserves three stop waits plus 0.25 seconds inside the wall budget,
including cold start; it closes input, sends TERM to the owned group and escalates
to KILL as necessary, then reaps the child. A natural child exit is a distinct
reason and cannot be represented as acknowledged model cancellation.

A separate evidence FD receives one aggregate receipt after cleanup: reason,
PID, exit code, child-reaped and group-alive flags, TERM/KILL flags, elapsed time,
and input/output byte totals. It never contains generated output. An undeliverable
receipt raises a fixed error, and unreaped children are explicit cleanup failures.
A reaped process group does not independently measure device allocator release;
actual host qualification still needs the relevant lock/process/memory observations.
SIGKILL of the supervisor itself cannot run its cleanup handler; the deployment
must retain the outer owner/host watchdog and must not advertise impossible
protection against host death or uninterruptible operating-system tasks.

## Verification scope

Regressions were observed before implementation: the actual Python open rejected
sequence one, and activation/supervisor modules were absent. Synthetic tests cover
complete-layer publication checks, hash/row/path/byte rejection before mutation,
serialization, persistent poison, receipt ordering, exclusive taps, EOF handling,
standalone child completion, ignored-TERM escalation, dropped controller, output
bounds and the real supervisor CLI. A full CLI fixture confirms that a supervised
child receives an activation on the separate socket and its next own request sees
the retained synthetic state, with final process-group and FD closure receipts.
All fixture arrays and text are public synthetic data; no model was run or deployed.

The next gate is explicit host staging of these exact modules and approved limits,
followed by one controlled own-turn/activation/continuation/close qualification.
SSH requires an on-host controller or explicitly forwarded private pipes/socket;
an extra local FD cannot simply be assumed to cross SSH. No MCDMA daemon, host
service restart, activation publication, held-out-item inspection or GPU work was
performed here, and model costs/energy remain unmeasured.

Baseline at `517b303e89568ed93dc62ef6f636847afaf40ffd` passed 424 reference tests
with three existing skips. Final focused worker tests passed 34 in 0.90 seconds;
full reference checks passed 441 with the same three skips in 19.81 seconds;
alternate-environment checks passed 155 with one existing sparse-indexer expected
failure in 7.73 seconds; plugin checks passed 29 and TypeScript compilation.
All Python commands used `PYTHONPATH=.` in the isolated worktree with the existing
primary `.venv` or `.venv-next` executables, and `git diff --check` passed.
Engineering mechanics are PASSED under these fixtures; actual host behavior,
model-side source attribution and native memory-release qualification are BLOCKED.
