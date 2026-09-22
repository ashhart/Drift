# Provider-owned terminal boundary

Local engineering gate: PASSED; actual OMP orchestration and native activation exchange remain separately unqualified.
Base commit: `335b9e98179cb0963f876282ceb1c8f873411bff`.
No hosts, models, network inference, translator fits, provider registration or production gate configuration changed.

## Opt-in interface

`workerProvider` accepts the optional lifecycle field below.

```typescript
beforeToolDispatch?: (boundary: Readonly<{
  identity: Readonly<Identity>;
  toolNames: readonly string[];
  signal: AbortSignal;
}>) => Promise<void>;
```

The hook runs after WorkerClient validates the real terminal reply, before toolcall_start, toolcall_end or done reaches OMP.
It also runs for a terminal without tool calls, with an empty names list.
Identity and names are frozen copies of the actual client's binding and validated tool events; argument values, text and activation data are absent.
The hook's signal combines the actual provider options.signal, WorkerClient lifecycle death/close, and the client's remaining absolute deadline.
There is no reliance on an OMP tool_call handler timeout or unsupported context.signal.

Earlier text continues streaming normally.
From the first tool call, the remaining event tail is held privately to preserve interleaved text/tool ordering; WorkerLedger's existing per-turn byte and event limits remain authoritative.
The tail never enters partial messages before successful release.
No-hook behavior remains unbuffered.

The provider checks cancellation, lifecycle state and remaining time again immediately before publishing the tail.
A hook rejection, cancellation, transport EOF, explicit client close or expired deadline aborts the transport and permanently poisons this provider instance.
Before that abort the provider now attempts one bounded graceful close, so a cancelled or rejected turn still ends with an ordered `close`/`closed` exchange instead of an EOF the worker cannot record.
The attempt is skipped when the client is already dead, is bounded by `closeGraceMs` (default 1000 ms) and never beyond the client's own remaining absolute deadline, and is always followed by the abort.
That default is a chosen prompt-cancellation budget, not a measured requirement; a close is a single transport round trip.
The graceful close adds no capability: it cannot dispatch tools, resume a parked tail or extend the turn, and independent process cleanup is still required separately.
A hanging hook cannot renew the deadline, and its late resolution cannot emit deferred calls.
Concurrent generation receives its own error without displacing or releasing the parked transaction.
The hook implementation must still invalidate any external readiness receipts on its signal; this change does not implement a file gate.

## Verification

Baseline plugin suite: 128 passed.
The initial regression run was red: six failures and one unchanged-default pass.
Focused provider/client/boundary tests passed24; the final full plugin suite passed137 and TypeScript passed.
The reference suite passed823 with three existing adapter skips in42.60seconds.
The next adapter environment passed155 with its known sparse-indexer parity xfail in8.77seconds.
These are local deterministic protocol/CPU tests, not native model or collaboration evidence.

Commands used from this isolated worktree:

```sh
bun test plugin/omp-drift/test/worker_boundary.test.ts plugin/omp-drift/test/worker_provider.test.ts plugin/omp-drift/test/worker_client.test.ts
bun test plugin/omp-drift/test
/opt/drift/plugin/omp-drift/node_modules/.bin/tsc -p plugin/omp-drift/tsconfig.json --typeRoots /opt/drift/plugin/omp-drift/node_modules/@types
PYTHONPATH=. /opt/drift/.venv/bin/python -m pytest -q
PYTHONPATH=. /opt/drift/.venv-next/bin/python -m pytest -q tests/test_adapters_next.py tests/test_translate.py tests/test_core.py tests/test_transport_sync.py tests/test_runtime_train_mail.py tests/test_properties.py tests/test_wire2_pool.py tests/test_hive.py tests/test_mailbox.py tests/test_m1.py tests/test_service.py tests/test_workspace.py tests/test_e3_compete_registry.py tests/test_adapters_mlx.py
```

Local evidence is retained at `/home/example/.codex/private-audits/drift-worker-terminal-gate`.

| Artifact | SHA256 |
| --- | --- |
| worker_boundary.ts | 6d87bd009388cb3ae1da7d89d19138cea5eace8119cab10e21e5fb9e7d98e9d8 |
| worker_provider.ts | c622651e8c529ea00b31afe23e763a66ab17fc16f1aaeb300653ddce0eebfce8 |
| worker_client.ts | ca9573998f9c70763f4f5608919fe3df61359ce55623f76009f72ec9f5a13e7f |
| red log | 0f51c753bbd524ffd2c2d878c1c37893658e5257f98f77157d155dba65b9f4c7 |
| plugin log | 9501117c990f6bb0b19ed7cf9bf0dca797887e8857de44be287b24b28d9bd1d8 |
| reference log | 777a24f7740b1f2993700873f902be7e0fca8de939d9790bbc62c9ebc0920e25 |
| next log | a68a936a7c2886304007998476d605c143313b39809961cf3fcead6805209f31 |

Next ticket: qualify the owner-pinned paired lifecycle wrapper using actual installed OMP with public synthetic workers, including real provider abort and stale-receipt rejection, before considering any native dispatch.
