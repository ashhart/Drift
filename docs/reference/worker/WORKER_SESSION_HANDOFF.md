# Private worker session interface

This ticket supplies a callable protocol core, an executable NDJSON dispatcher,
an exact-prefix checkpoint codec, and a real Studio continuation operation.
Synthetic protocol and native-state wiring tests are engineering evidence only;
they do not establish a running GLM/Qwen OMP coding pair or source ownership.

## Concrete inspected seams

`studio_drift_worker.py` keeps `S['cache']` across commands, but its original
`generate` stopped at a boundary token without consuming it and latched `done`.
The new explicit `continue` command takes private own-token IDs, consumes the
pending boundary followed by those IDs, retains the same cache, and clears the
latch only after success. Invalid input is rejected before mutation; errors during
consumption poison the session. Existing `extend` behavior and default response
shapes remain unchanged. `generate` now records the unconsumed boundary privately.
These token IDs are worker-own input, never activation-channel payloads.

`CheckpointCodec` calls the supplied checkpoint tokenizer's real
`apply_chat_template` with tools and thinking disabled. It verifies that each new
render begins with the exact previously consumed prompt, generated IDs and stop
boundary. A rewrite fails with CAPABILITY rather than resetting the native cache.
The assistant message and exact generated IDs must come from the same worker;
re-tokenizing decoded output is not accepted as proof of its native history.
The caller passes only the returned suffix to `continue`, whose pending boundary
is separately consumed once. Model-specific tool parsing and actual checkpoint
template continuity must still be qualified; this module invents neither syntax.

The Spark runner currently opens one completion request with a fresh cache salt;
its connector removes per-request live metadata on finish. Its existing public
completion seam therefore provides no established retained-cache tool turn.
A separately labelled reconstructed implementation may prefill its own canonical
transcript and re-inject pinned memory, with every repeated input token charged;
that is distinct from native-state retention and belongs to the GLM backend.

## Backend and executable contract

Run explicitly with `python -m drift.serving.worker_stdio --config CONFIG
--backend module:factory`. The private configuration has `worker`, `pins` and
`limits`; `factory(config)` returns the backend. Loading this module alone starts
nothing. A production factory must verify local model and translator files against
the approved pins, own its process/native state, supply the actual tokenizer and
parser, and bound blocking backend calls. The interface's pin comparison is not
hardware attestation or artifact verification.

Backend attributes are `backend` with value `live` or `fixture`, and
`capabilities` containing `native_state` as `retained`, `reconstructed` or `fixture`,
plus booleans `tool_calls` and `cancellation`. Methods are `open(payload)`,
`own_prompt(payload)`, `tool_result(payload)`, `stream(max_tokens)` yielding events,
`cancel(target_seq)` and `close()`. Cancellation must unblock the stream iterator;
the protocol acknowledges only after that iterator exits. It does not equate an
iterator exit with server-memory release. Backends must preflight actual token
limits before native mutation and bound their calls; checking terminal accounting
cannot retroactively prevent an oversized prefill or interrupt a stuck kernel.

Every frame has exactly `v`, `session`, `worker`, `seq`, `op`, `payload`; version
is 1 and command sequence starts at one and increments. Replies echo identity and
command sequence. Own text and tool results pass only over this private channel.
Opened replies bind model identity, model/translator hashes, accepted limits and
truthful capabilities. See `worker_session.py` and provider integration for the
agreed open/own_prompt/tool_result/stream/cancel/close payload definitions.

Limits bound individual input JSON bytes, emitted session bytes, output events,
turns, per-turn output tokens, total reported input plus output tokens, and elapsed
time. Every reconstructed input is counted again. The dispatcher caps NDJSON
lines and keeps one active stream; EOF cancels and closes its backend, with bounded
thread joining. A supervising process still needs a hard wall timer and verified
owned-child cleanup for a backend that hangs during close or kernel execution.
Cancellation terminates the protocol session; callers must use fresh identities.

The dispatcher is intended for a caller-owned private pipe under worker-specific
OS identity and filesystem isolation. A worker/session string is binding data,
not authentication. This ticket creates no public socket, permission boundary,
remote process, GPU job, activation export or test-set access. Private paths,
credentials and configuration must be supplied by the deployment controller.

## Verification and next gate

The original real-worker handle regression failed with `unknown op continue`;
protocol tests failed before the new modules existed. Baseline reference tests
were 407 passed and three pre-existing skips. Tests cover pin mismatch before
backend use, worker and sequence mismatch, oversize input, sanitized errors,
usage bounds, actual Studio handle wiring, retained cache, deferred stop tokens,
poisoning, exact template prefixes and cancellation after iterator termination.
The next gate is the concrete backend factory and installed OMP provider contract
under the approved local isolation, followed by bounded actual-worker qualification.

## Concrete Studio route and artifact preflight

`python scripts/live/studio_drift_worker.py --worker-session-config CONFIG`
runs the existing guarded model loader and then the private NDJSON dispatcher,
suppressing the legacy ready event in this explicit mode. `StudioBackend` routes
own prompts and tool results to native continuation and consumes one generation
token per cancellable iteration. It holds bounded turn output until the installed
tool parser finishes; this first adapter does not promise partial-text streaming.
Multiple tool results are accumulated and rendered together before the next turn.
Native output IDs and the actual deferred stop come directly from `generate_own`,
which is a private operation; the existing generate response stays unchanged.

The loader uses `AutoTokenizer.from_pretrained` with `local_files_only=True` and
`trust_remote_code=False`. Tool parser selection follows the inspected oMLX
`engine/vlm.py`: `mlx_vlm.tool_parsers._infer_tool_parser`, `load_tool_module` and
the selected module's boundary markers and `parse_tool_call`. Final extraction
uses the existing `omlx.api.tool_calling.parse_tool_calls`. These are inspected
runtime APIs, but the installed Studio versions and checkpoint prefix behavior
still need the actual-worker qualification. No parser fallback is fabricated.

Both CLI routes call `verify_worker_manifests` before backend dispatch; Studio
performs this before model loading. Configuration provides `model_manifest_path`
and `translator_manifest_path` plus their expected digests in pins. Each manifest
uses the existing recipe's `artifacts` and `sha256` maps. Relative paths resolve
beside the manifest; canonical absolute paths are accepted; symlinks are rejected.
Artifact hashing is bounded to 64 MiB total by default or the explicit configured
`max_verified_artifact_bytes`. Model manifests must declare `weights_verification`,
explaining weight bytes not independently hashed. Studio also requires its exact
canonical `checkpoint_path`; configured manifest digests alone do not establish
full checkpoint-weight verification or runtime qualification.

Cooperative Studio cancellation is acknowledged only after the decode iterator
ends; it cannot interrupt a stuck native kernel, and host-level hard deadlines
remain required. CLI fixtures and command-adapter fixtures ran without model
weights, remote operations or private activations. GPU costs and energy are
unmeasured, not zero-valued live-run measurements.

The legacy 15-minute Studio idle-exit guard remains active in the explicit worker
mode; the controller must impose its shorter declared wall limit. The subsequent activation-only integration and owned supervisor are documented
in WORKER_ACTIVATION_HANDOFF.md; actual host qualification is still required
before calling this a linked Duo-drift run.

Focused final checks passed 17 tests in 0.09 seconds; reference full checks passed
424 tests with three existing skips in 19.68 seconds, alternate environment checks
passed 155 tests with one existing sparse-indexer expected failure in 8.21 seconds,
and plugin checks passed 29 tests plus TypeScript compilation. All commands used
the isolated worktree's PYTHONPATH and the existing primary virtual environments;
no dependencies were installed and no model inference was launched.
