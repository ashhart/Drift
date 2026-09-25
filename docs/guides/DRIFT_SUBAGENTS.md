# KV-only OMP subagents

`/drift subagent enable` selects a private, owner-configured parent/child pair;
it does not convert a stock Duo room or provision inference servers and MCDMA.
The installed-OMP synthetic check passes two repeated exchanges and ordered
worker closure. Native qualification of this new KV-only command remains
**BLOCKED**: the latest attempt completed no exchange while the configured
GLM service was unavailable. Earlier native runs used a different, text-enabled
control policy and do not qualify this one.

## Channel boundary

Each model receives its own initial instructions. The parent can call `task {}`
once, followed by `drift_sync {}`; the child can call `drift_sync {}` and finish
with `yield {}`. Extra arguments, peer prompts, Hub messages, file tools and
arbitrary nested agents are rejected. The owner supplies the child instructions
in its `drift-peer` agent definition, separately from the parent's task.

The wrapper delegates task admission using fixed lifecycle strings and replaces
the tool result with `ready`; it never forwards model-generated task arguments
or a child's answer. Fixed task-control strings still exist outside the
activation channel and must be counted. A child finishes with fixed
`{"complete":true}` metadata, not its answer. Once both routes settle, the
context hook aborts any automatic OMP follow-up before the provider sees it:
OMP otherwise inserts a background-task completion as a developer message.
Unexpected text during the active exchange still poisons the provider.

Only KV tensors and bounded metadata use the designated MCDMA activation path.
SSH may launch workers and carry each worker's own input/control, not its peer's
memory. Do not describe setup prompts or fixed control strings as zero text.

## Owner setup

Use a dedicated OMP process and a fresh, empty session. Keep profiles, activations
and receipts in owner-only directories outside the agent task directory and all
repositories; enforce filesystem isolation when testing hidden/private tasks.
Link the full checkout's `plugin/omp-drift` directory, not a copied `src` folder,
because the extension also uses the repository's `scripts/omp` modules.

The owner prepares and hashes these profiles, then supplies their absolute paths
and SHA-256 values before starting OMP:

| Path variable | Digest variable | Required contents |
| --- | --- | --- |
| `DRIFT_WORKER_CONFIG` | `DRIFT_WORKER_CONFIG_SHA256` | Exactly two linked, fixed-session, multi-turn workers with `communication_mode: "kv_only"` and distinct `subagent_role: "parent"` / `"child"` |
| `DRIFT_PROVIDER_BOUNDARY_CONFIG` | `DRIFT_PROVIDER_BOUNDARY_SHA256` | Version 2, matching actors, private control root, canonical task root, fresh nonce, absolute deadline and bounded epoch count |
| `DRIFT_EXCHANGE_CONFIG` | `DRIFT_EXCHANGE_SHA256` | Version 2, private coordinator socket, both bound routes and `delivery: "next_turn_snapshot"` |

See the [native owner guide](NATIVE_OWNER_EXCHANGE.md) for the worker/socket
contracts and [exchange modules](../reference/exchange/EXCHANGE_MODULES.md) for
the staged-delivery protocol. The maximum boundary lease is 170 seconds; it is
not extended by repeated calls. Set `tools.intentTracing: false` and disable
compaction in this process's private OMP configuration, since extra tool text
and history rewriting violate the pinned KV-only contract.

Start fresh, exclusively leased Spark bridge consumers before pinning their
mailbox sessions. A replacement producer refuses nonempty old ACK/header slots;
do not erase these records to bypass the guard. Existing MCDMA daemons alone
are not sufficient. Confirm inference health, deployed connector hashes and
host ownership before a native run; an earlier health check can become stale.
SSH worker commands must explicitly include any required `SSH_AUTH_SOCK` in
their pinned environment; the provider does not inherit ambient credentials.

The coordinator now has a bounded stdio entry point on its owner host:

```sh
python -m drift.exchange.bootstrap \
  --profile /private/drift/coordinator.json \
  --sha256 "$COORDINATOR_SHA256" --seconds 120
```

Keep stdin open for the lease; `{"op":"shutdown"}` or EOF closes this
coordinator, and `closed` is emitted only after its cleanup returns. `ready`
means the coordinator is listening, not that models or transfers are qualified.
This process does not own or restart the shared MCDMA verbs daemons.

In the configured OMP session:

```text
/drift subagent enable
/drift subagent status
/drift subagent disable
```

Enable verifies fresh route cursors and selects the pinned parent model; then
give that parent its own task. Paired staged receipts release each epoch, and
subsequent boundaries confirm actual application. The two terminal boundaries
confirm the final pending publications before settling. Disable closes the
workers, stops release and puts back the model and tools the session had before
enable, even when a worker does not close in order; its notice says which
happened. Independent native termination receipts remain necessary to claim
resource cleanup. A refused command, such as a second enable, a mistyped one or
a disable from another session, changes nothing and leaves the pair running. A
failed or completed pair is not reusable: start with fresh sessions, profiles,
mailbox consumers and lease.

## Reproduce local integration QA

On macOS with the supported OMP installation, this command uses deterministic
stdio workers and a private Unix socket, with network access denied except Unix
sockets; it does not load models or exercise RDMA:

```sh
python scripts/omp/kv_subagent_probe.py --omp /path/to/omp \
  --source "$PWD" --production-command
```

It calls the actual `/drift subagent enable` entry point, delegates one real OMP
task, runs two paired exchanges, waits after terminal completion to catch the
async text-delivery race, and requires exact worker-call counts and closure.
The report explicitly identifies synthetic versus native evidence.
