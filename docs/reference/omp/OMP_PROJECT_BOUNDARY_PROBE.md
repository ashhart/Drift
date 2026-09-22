# Public OMP project-boundary qualification

Status: FAILED for the candidate tool-call boundary; native project dispatch remains BLOCKED.
This ticket adds only a public fixture/probe and verdict tests, not a production boundary fix.
The candidate files were copied, hashed and tested without modifying their parent worktree.
Stock Duo was copied unchanged; no models, network, SSH, native activations or private model outputs were used.

The `project_boundary_*` candidate is retained to reproduce this failure and is
not registered by the native API runner. That runner still rejects
`activation_supplement`. Do not load the candidate extension for native workers:
the route helper is separate and the tool-call hook has no usable cancellation
signal. Its unit tests qualify only shared waiting and file publication, not a
safe project exchange.

## Exact executable and candidate

OMP executable SHA256: `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a`.
Stock Duo command SHA256: `ca699d1e07b273da2d95dac696c97d71d05aaa08260ea5fba91782097cf89e21`.
Candidate control: `a151a721d168ba1f04fe978c5ebbffb0ee1cf8f4c5a974037f55ebbf09a903d1`.
Candidate extension: `86e950183d4dd663bde5f0949cf16eeb41700dba916cd1d8367d460653d9e6f0`.
Atomic candidate files helper: `5ca41a1a17fc4db9cd39521e80a62ff905542c0aaf2c057b6569442b8ffbae29`.
The timeout observation preceded the atomic helper update and used helper hash `36448123311ab9344129312d35ee0f94817c62225c7cfea217f6923e44b3172d`; the control and extension were identical.

## Actual observed behavior

Both sessions shared one Bun PID while their private Python worker sessions remained separate.
The parent task launch completed before either boundary parked.
The fixture deliberately delayed every terminal event by 200 ms after its tool-call payload.
Every observed OMP tool_call hook still saw terminal_count equal to stream_count; the narrow post-generation ordering hypothesis passed.
The release run published both readiness records with complete worker generation, admitted no Hub tool before the bound releases, then closed both workers exactly once in 3.425 seconds.
One release pair admitted subsequent tools without requiring another exchange.

A pinned isolated `agent/config.yml` with `extensionHandlers.toolCallTimeoutMs: 150` reproduced the failure in 2.961 seconds.
OMP blocked the timed-out child Hub call but resumed the child's next generation while child-ready.json remained.
When parent readiness arrived, child terminal_count was 1 and stream_count was 2.
The delayed bound release subsequently admitted a parent Hub call against the stale child receipt.
Thus a validly hashed receipt alone does not prove the peer is still parked.

Invalid-release and deadline failures blocked all Hub admissions, but OMP recovered and exited zero; neither run removed stale readiness.
Group SIGTERM while both actors were parked blocked tools and reaped the OMP leader with exit143, but both worker close counters stayed zero and readiness survived.
This probe does not claim graceful native worker cleanup or independently prove that all descendants were reaped.

Actual tool_call event keys were input, toolCallId, toolName and type.
The actual event.signal, context.signal and context.abortSignal were undefined.
Adjacent runner.ts lines215–228 and1278–1283 expose only an internally signal-bound UI facade, not a general handler cancellation signal.
Its lines1457–1480 use the observed timeout setting and block handler failures, but that does not cancel candidate-local pending promises.
No unsupported signal property was invented.

## Evidence

Public synthetic reports are preserved outside the project at `/home/example/.codex/private-audits/drift-boundary-public-20260921`.

| Report | SHA256 |
| --- | --- |
| drift-boundary-release-delayed.json | 8743b91c7201a08a73e1fb4b80bb6ea21e53c6146e8ad017ae0bf746e9cf3d3c |
| drift-boundary-timeout.json | de5bec282fd430815642b4e5832aa9e15040cdfde583663490669747b35c2afc |
| drift-boundary-invalid.json | 5de03a30efc42ad99b0515c05a1aa82582604972234bbe0d0cac9c1e52a8c200 |
| drift-boundary-deadline.json | 8c87e9d19b9b6ead7158a52164f91ebb59bcf20c00a98646ba21eb69570f4697 |
| drift-boundary-cancel.json | 670135cdff7b47b8dc179892be1922cd951fda3bd22045289ea86a2d042af3e9 |

The baseline actual Duo fixture passed before adding probes; focused verdict tests passed4 and the full reference suite passed821 with three existing adapter skips in41.55seconds.
All costs are CPU fixture wall time; no model tokens, model energy or native resource costs were measured.

## Reproduction and next gate

Run `scripts/omp/boundary_probe.py --omp /home/example/.local/bin/omp --source SOURCE --candidate CANDIDATE --duo DUO --mode release` using the project's existing Python environment.
The candidate must contain the three project_boundary modules and paused_echo_files.mjs; the probe does not bundle or substitute them.
Modes timeout, invalid, deadline and cancel exercise failure handling and return exit2 when the boundary does not qualify.
Reports contain only public fixture metadata and hashes; subprocess text is counted and discarded.

The next bounded proposal is provider-owned parking before tool-event emission, after actual worker terminal, using the provider's real options.signal and explicit owner deadline.
That proposal needs its own failing cancellation and cleanup regressions before implementation or native dispatch.
Raising the OMP handler timeout alone would avoid one deadline ordering but would not establish cancellation ownership or eliminate stale readiness.

Parent review added two red regressions before tightening the probe verdict:
missing worker records cannot count as a complete pair, and removing readiness
does not excuse absent worker cleanup in a cancellation trial. The resulting six
verdict tests pass, and reassessing the five preserved reports above changes none
of their outcomes. The raw evidence was not rewritten.
