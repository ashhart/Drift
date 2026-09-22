# Single-publication preparation gate

This local engineering ticket supplies a request-start gate and a bounded delivery
harness. It does not authorize or record a model run. The already qualified no-link
GLM cancellation path remains unchanged; existing callers do not accept release.

## Request and publication order

Start the runner with `--control-stdin --wait-publication --ready-timeout N --no-tap`.
It creates fresh head inbox/outbox folders, refuses any existing session evidence,
tokenizes the local prompt, and emits exactly `{"publication_ready":true}`. It sends
no completion request until one exact `{"op":"release"}` control frame arrives.
Abort or controller EOF stops the owned child while waiting. Timeout, early release
and duplicate release fail closed. `--no-tap` sets the deployed connector's existing
`drift_tap=False` option, so this one-way request exports no outbound activations.

The controller must observe readiness, stage the peer's sequence-zero file and a
hidden head file, and verify every rank's identical digest before publishing the
head file and sending release. `SinglePublication.run(stage, publish_and_release)`
enforces that order around the callbacks using one deadline and returns the verified
rank receipts. Both callbacks receive their remaining timeout and must respect it.
`RankDelivery.health`, `admit` and `wait_applied` accept that absolute deadline;
`wait_applied` returns validated fixed-schema receipts instead of discarding them.
The controller owns request cleanup in a `finally` block on every outcome; the
publication harness cannot certify process shutdown and explicitly reports it BLOCKED.
No-link mode invokes neither publication callbacks nor rank transport. A gate is
single-use even after failure, and admits at most twelve rows in sequence zero.

## Proposed minimum live workload, pending dispatch

Use one public synthetic coding fixture's final Qwen prompt token as the source,
translated with `configs/frozen.live-qa-v3.json`'s reverse v3 artifact, verified SHA-256
`36c352797d936aca6468d4066c1be93fea99ab0185f7505e731874dc5e7be8b9`, gain one and twelve
copies. This yields one twelve-row publication across eleven GLM layers, each row
containing 512 latent values. The exploratory loop's older reverse defaults cannot
silently substitute. Keep raw taps and publication bytes in private staging outside
model-readable workspaces and return only hashes, dimensions, timings and receipts.

Proposed GLM limits are one request, at most 256 generated tokens, at most 256 own
prompt tokens including twelve reserved entries, one publication and a 180-second
outer deadline with cleanup time reserved. These are proposed caps, not authorization
or measured costs. Use fresh unique folders on both ranks and retain all failures.
The existing cancellation `OwnedSession` deliberately rejects linked commands and
the readiness event: its no-link contract must not be bypassed to run this gate.
The explicit `LinkedSession` sibling and `scripts/qualify_live/linked.py` now integrate
this protocol without widening the no-link client contract; live dispatch remains separate.

The Studio worker requires its existing admission guard, currently 150 GiB estimated
need plus a 48 GiB reclaimable reserve, MLX limit 158 GiB and cache limit 4 GiB; these
are configured estimates and limits, not GPU guarantees. It also executes a four-token
startup cache probe, which must be counted. Its current quit command cannot interrupt
an in-flight MLX call, and observing a local SSH exit cannot certify that its remote
worker stopped. Owned on-host Qwen supervision and shutdown evidence remain required
before a two-worker linked qualification can pass.

## Evidence limits

Post-write receipts establish successful local writes for the declared digest on
every rank; they do not establish atomicity or semantic use. The connector applies
memory after an engine step's logits have been computed. Pre-staging before request
start does not establish that every output token followed observed receipts, a
prior-epoch causal barrier, provenance preservation, or an OMP coding benefit.
Report initial load/write costs separately and do not hide them behind warm-up.
