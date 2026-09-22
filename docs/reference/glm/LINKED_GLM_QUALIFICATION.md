# Runnable one-way linked GLM qualification

This prepared engineering driver consumes one privately staged translated Qwen
publication. It does not create the source fixture, load Qwen, qualify its shutdown,
or claim source ownership, causal use, atomic TP writes or coding improvement.
No linked live run is recorded by this document.

Run `scripts/qualify_live/linked.py` on the head serving host with `--config`,
`--runner`, `--ledger`, `--report`, `--publication` and `--publication-sha256`.
The initial proposed profile is `configs/qualification.linked-glm-v1.json`.
Before dispatch, independently verify/freeze the staged sources, profile, invocation
and controlled source fixture's hash; verify head-to-peer SSH host identity/access,
the active connector receipt, and idle resources. This driver grants no new resource
authority and performs no retries, restarts or warm-up.

The fixture is checked for the exact eleven `l3` through `l43` layers, twelve rows
per layer, 512 values per row, finite numeric values, a 1 MiB compressed-file cap,
and the expected SHA-256. Its expected source is the already documented frozen
reverse-v3 recipe, but this driver cannot infer or authenticate source provenance
from an array; retain the source-generation receipt separately.

The driver requires three complete idle target metrics before one request starts.
Its explicit `LinkedSession` sibling requires capped generation/prompt flags,
`--wait-publication`, `--no-tap` and supervised control. Prompt token count is checked
inside the runner before readiness or the completion POST, including reserved rows.
The existing `OwnedSession` constructor still rejects this linked command.

After readiness, one private sequence-zero file is staged to a fresh peer folder
and a hidden head file. Every rank's digest must match before the head file is
published and typed release is sent. Local hard-link publication refuses replacement
of an existing final file; peer staging likewise refuses any existing session folder.
Every rank must acknowledge the same digest, twelve rows, sequence and world size;
errors, missing receipts, timeouts and natural completion cannot be accepted as a
successful cancellation qualification. No raw model output is retained or printed.

Only after verified delivery plus owned output activity and target running=1/waiting=0
does the driver send scoped abort. It requires the typed cancellation event, expected
supervisor exit, EOF/reaping, and three subsequent zero-running/zero-waiting samples
with KV occupancy at baseline. The work deadline is 170 seconds and the outer limit
180 seconds, reserving cleanup time. OS launch/termination and storage still have
operating-system dependencies; late or unconfirmed cleanup cannot pass.

The report preserves exact rank receipts, hashes, cumulative publication duration,
request/output counts, process evidence, metrics observations and whole-probe time.
Publication duration includes staging and polling rather than isolating GPU write
latency. Generated-token count and energy remain unmeasured; the token cap is not
measured use. Failures preserve staged evidence and never trigger an inference retry.
The report explicitly leaves Qwen shutdown BLOCKED and semantic/causal use NOT_TESTED.

Local regressions run the actual gated runner, linked client, publication staging,
rank probes, cancellation observer and metrics fetch against synthetic HTTP/SSE and
rank-filesystem fixtures. A successful simulated write receipt is not a real cache
write result. Missing peer receipts, natural completion, early/duplicate release,
unsafe scope, oversized prompts, deadline expiry and evidence replacement fail closed.
