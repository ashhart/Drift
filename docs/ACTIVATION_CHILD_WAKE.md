# Receipt-bound fresh-child wake

This local candidate implements one lifecycle subset, a fresh Studio worker
whose native prefix has not yet been consumed. It does not implement generic
stopped-worker resume or change the installed OMP/Pi subagent launchers.
Checkpoint-level semantic qualification remains BLOCKED.

The owner configuration must opt in with `experimental_activation_wake: true`
and configure the existing activation receiver. The flag defaults to false.
Opening the worker pins the checkpoint-rendered local system/assistant bootstrap
without evaluating it. That setup is not supplied by the peer wake message.

After the activation receiver applies and settles the foreign cache, it records
the exact session, target worker, activation sequence and cumulative foreign-row
count under the same native-state lock. The controller then sends:

```json
{"op":"activation_wake","payload":{"activation_seq":3,"foreign_total":4}}
```

The operation uses the existing worker envelope and returns
`activation_wake_ack` with exactly the two integers. This acknowledgement only
arms the next bounded `stream`; it is not a semantic-use receipt. At stream
time the worker checks the binding again, renders and checks the pinned local
bootstrap, reserves its token budget, then consumes it and computes the first
generated token while holding the native lock. The generated first result is
used once, and all local bootstrap/output tokens are counted. No consumed token
is replayed to refresh stale logits.

Wrong or changed receipts, extra fields, task text, prior native consumption,
stopped sessions, pending tool/control state and inadequate budgets fail closed.
An additional append between arming and streaming invalidates the receipt.
The normal worker's own-input operations still exist as separate authorized
channels; this option does not turn their presence into a strict KV-only policy.

The OMP worker client exposes `activationWake(appliedReceipt)`. It requires the
complete existing `appended` receipt, validates exact fields and the child's
session/target, then transmits only its numeric sequence and row count. It
rejects wrong acknowledgement fields or bindings. It never calls `ownPrompt`,
`session.prompt` or a provider fallback.

The installed OMP task executor still launches with `session.prompt(task)`;
the installed Pi subagents still use task prompts and `sendUserMessage` for
steering. Those paths were inspected, not modified or replaced. No claim of
automatic KV-only OMP/Pi subagents follows from the new worker/client operation.
The controller still needs an owner-pinned activation-only launch route and a
qualified way to deliver the parent's memory to the child. Wake control does
not establish that this delivery used RDMA, nor does it add a transport.

The red/green regression exercises the actual Python session, backend, native
dispatch and activation control with synthetic caches and a deliberately
memory-dependent fake compute boundary. It proves ordering and accounting, not
natural-language recall. Native checkpoint first-token causality and recurrent
state parity remain separate gates, as does any resumed-child lifecycle.
