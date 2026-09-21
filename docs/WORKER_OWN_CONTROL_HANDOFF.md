# Explicit own-control updates during bounded multi-turn sessions

The default worker behavior is unchanged. Setting the owner configuration's
`experimental_multi_turn` to true explicitly enables `own_control` in both the
generic worker CLI and the concrete Studio route; the OMP client separately owns
its opt-in lifetime/idle deadline and default close-on-stop behavior. This flag is
an experimental lifecycle choice, not a new live qualification or capability claim.

Private v1 frames use `op:"own_control"` and `payload:{"system_prompt":[...]}`;
the reply is `own_control_ack` with an empty payload. They never cross the activation
socket and do not contain foreign-memory metadata. The worker rejects unsupported
backends, absent opt-in, malformed snapshots, active/prefilled turns, session byte
or wall violations, and more control updates than its fixed maximum turn count.
The open reply's exact capability shape remains unchanged.

A backend implements `own_control(payload)`. The shared
`worker_own_control.own_control_message` helper represents the new snapshot as an
explicit new system-role message, saying it supersedes earlier own-control
snapshots only. Historical own user/assistant/tool messages are preserved. Empty
snapshots explicitly carry no additional own-control instructions; they do not
pretend the earlier native state or base system context was erased.

Studio queues the newest unconsumed snapshot, then appends it before the next own
user message or after the complete pending tool-result group. No model work happens
at the control acknowledgement itself. Exact checkpoint tokenization and prefix
validation run when that next group is consumed, and all new input tokens are
charged then. Multiple tool results cannot cause repeated partial prefill of the
same group. Replacing an unconsumed queued snapshot does not alter native history.
A template that rewrites the consumed prefix fails CAPABILITY; there is no hidden
cache reset, prefix reconstruction or token-text re-encoding fallback.

A reconstructed GLM backend may use the same helper with its own canonical
transcript, preserve every previous turn and charge the full repeated prefill on
every generation. That backend's implementation remains separate from this
retained-cache Studio ticket. Neither path proves that model behavior matches a
freshly rebuilt system prompt or that source ownership is understood; actual
bounded multi-turn and tokenizer behavior still require host qualification.

Four initial regressions failed before implementation. Tests cover default-off and
missing-backend rejection, two own turns with cumulative usage, control changes
only between turns, control-update bounds and malformed payloads, unchanged native
cache/history, grouped tool results, and fail-closed template rewriting. Fixtures
use no checkpoint inference, remote mutations, private activations or held-out
items. This is an engineering implementation, with model cost and energy unmeasured.

Baseline at `98fbe13d0c72e14c64985f4c2c1d25cd136488ff` passed 441 reference tests
with three existing skips. Final focused checks passed 25 in 0.11 seconds; full
reference checks passed 448 with the same three skips in 19.65 seconds; alternate
checks passed 155 with one existing sparse-indexer expected failure in 7.87 seconds;
plugin checks passed 29 and TypeScript compilation. Commands used the isolated
worktree's PYTHONPATH and the existing `.venv`/`.venv-next` executables, with no
installation, inference or remote work; `git diff --check` passed.
