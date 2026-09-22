# Installed OMP provider contract probe

Engineering verdict: PASSED for provider registration and one synthetic tool round trip in the installed OMP executable; real GLM/Qwen integration remains BLOCKED.
Verification HEAD: `a9be14ef5b52ea9502d01bd69d916b73b2f6be80`.

## Installed entry point

`/home/example/.local/bin/omp` resolves through `/home/example/.bun/bin/omp` to `/home/example/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js`. Despite its suffix, that file is a compiled arm64 Mach-O Bun executable reporting `omp/18.2.6`. The adjacent package metadata and source report `18.0.7`; they are not the executable's implementation.

Read-only inspection of the executable's embedded source located `packages/coding-agent/src/extensibility/extensions/loader.ts` at byte 141748823, the extension runner at 141758848 and `config/model-registry.ts` at 143906676. The factory API exposes `registerProvider(name, config)`; the runner forwards it to the model registry with the extension path, and the registry requires `config.api` when `config.streamSimple` is supplied. The executable's `omp:legacy-pi-shim` resolves bundled OMP imports. The live probe below verified those mechanisms through the executable rather than relying on the stale adjacent declarations.

## Boundaries and command

```sh
.venv/bin/python scripts/omp/probe.py --omp /home/example/.local/bin/omp
```

The probe creates a temporary configuration and working directory, copies one fixture extension there, then launches the installed executable through `/usr/bin/sandbox-exec`. The sandbox denies all network access, all writes outside that temporary tree, and home-directory data reads except installed runtime dependencies and native modules. The child receives an explicit environment allowlist without inherited provider credentials; both `PI_CONFIG_DIR` and `PI_CODING_AGENT_DIR` resolve to the temporary tree. User configuration and sessions are not modified. The first attempt set only `PI_CODING_AGENT_DIR` and was blocked when OMP attempted global daemon-presence creation; setting both directory controls fixed that isolation issue without loosening the sandbox.

The command disables ambient extensions, skills, rules, built-in tools, session persistence, title generation, PTY and LSP. Its only provider is an extension callback that emits two fixed event sequences from JavaScript; its only tool returns a fixed string without I/O. OMP processes that synthetic transcript through its actual provider, tool execution and result-delivery loop. No language model, model server, training task or repository work is involved. The child has a 20-second session cap and a 40-second subprocess deadline; the callback refuses a third invocation. Captured transcript/error content is not printed, only hashes and a fixed error category.

## Observed evidence

The final executable probe exited zero in 0.474 seconds with these exact fixture counters:

| Field | Observed |
| --- | --- |
| Provider registration completed | true |
| `streamSimple` calls | 2 |
| Extension tool executions | 1 |
| `tool_call` extension events | 1 |
| Matching `toolResult` messages seen in the second callback | 1 |
| Other tools invoked | 0 |

This establishes that `AssistantMessageEventStream` imports correctly inside this executable, a registered synthetic provider is selectable, its tool call enters OMP's execution path, the tool event hook fires, and the tool result reaches the next provider context. A static shape check or mock host alone would not establish that. It does not establish real inference, streaming backpressure, parallel worker isolation, cancellation, cache ownership, source attribution or a usable repository workflow.

| Artifact | SHA-256 |
| --- | --- |
| Installed executable | `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a` |
| `scripts/omp/probe.py` | `aefb3dbc877fa8840b1176875837f2bfcae96b0ef572d6aafae718663a76a163` |
| Fixture extension | `78e34ae4e4f8ecfffb1efc7a7c35c9f77e1c42eb95644bd51b4573e89aa1b1cc` |
| Probe tests | `a8fbaa9707432ddeafadba3de96b4d7f9842f5d1876fd013d584d6ccb3d170f0` |
| Final sandbox policy | `431ddbd1ec2a75fdc6337a583b2ebc95668b4f4110dabed07bd4804e97f8e81f` |
| Final synthetic stdout | `2cbb2534cc8d16b4854183087b108d0d9c4250fa155cf78c7023a37f38568f5b` |
| Final stderr, empty | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Temporary paths and event timestamps make policy/transcript hashes run-specific. The probe removes temporary files on exit; these are aggregate receipts, not retained model evidence. GPU time, token costs and energy were not measured; the provider fixture performs no inference and reports zero usage. The pre-change service baseline passed six tests; the new contract tests were observed failing before implementation and then passed seven tests.

## Next minimal adapter task

Implement a private worker-session client that binds one OMP session to one qualified model worker and pinned translator manifest, then maps actual assistant text, tool-call and terminal events into the now-tested `AssistantMessageEventStream` contract. Specify and test the required session-open, tool-result, stream and cancel operations before claiming a live endpoint exists. The worker's own prompt and tool-result traffic is a separate authorized local-input channel; it must never be smuggled into the cross-model activation transport or the current strict lifecycle protocol. Follow with a bounded actual-worker qualification covering tokenizer/chat templates, native state continuity, failure propagation and acknowledged cancellation under approved resource limits; source ownership still needs its own controlled evidence.

Full verification command `./scripts/check_all.sh` completed successfully after the implementation: reference environment 247 passed and two existing skips in 17.98 seconds; next environment 155 passed and one existing sparse-indexer parity xfail in 7.94 seconds; plugin 29 passed plus TypeScript checking. The missing-MLX and alternate-Transformers skips do not qualify real adapters, and the existing xfail remains unresolved. `git diff --check` also passed.
