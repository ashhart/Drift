# OMP live integration

Status: BLOCKED for the real GLM/Qwen repository workflow; PASSED for the local reference-start guard and plugin regressions only.
Base commit: `9cad535084c6fdf5abecbb06d2a8ed9856463b84`.

## Architecture and inspected source

`omp-drift` is a separate OMP plugin controlling a stateful model runtime. Duo is the existing text-based comparison workflow. MCDMA belongs underneath the runtime as a transport backend after qualification; it does not supply model sessions, source ownership or an agent tool loop.

The inspected Duo checkout is `/home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo`, commit `8aeb53d3377928874e364bc6f6f8a05b471983cc`, linked from the installed OMP plugins directory. Its `src/command.ts` opens a room by selecting a model, configuring a peer override and sending a next-turn instruction to dispatch the `duo-peer` task agent; it requests cancellation through the Hub. This is text-based agent orchestration, not access to another model's native cache.

The local OMP executable reports `omp/18.2.6`. Read-only resolution follows `/home/example/.local/bin/omp` through `/home/example/.bun/bin/omp` to the global package's `dist/cli.js`, a compiled arm64 Mach-O Bun executable, SHA-256 `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a`. The sibling package metadata/source report `18.0.7` and do not identify the running implementation. Inspection of embedded source in the actual executable confirms the extension factory's `registerProvider(name, config)` entry point, forwarding through the runner into `modelRegistry.registerProvider`; a provider supplying `streamSimple(model, context, options)` must also specify `api`. Embedded source offsets were 141748823 for `extensibility/extensions/loader.ts`, 141758848 for `runner.ts`, and 143906676 for `config/model-registry.ts`. The binary has a bundled module resolver, so disk-only resolution is insufficient. This resolves the source-discovery discrepancy, but no runtime extension contract smoke test or model session was launched.

The MCDMA checkout inspected was `/home/example/Documents/Projects/Personal/Apple-Foundation/MCDMA-release-prep`, commit `0641a4866806cbb712f7e3aad347fb9741364bdb`; its README explicitly assigns buffer allocation, registration lifetime and GPU synchronization to the inference integration. The native client source and `SERIES.md` at `/home/example/Documents/Projects/Personal/Vontra/benchmarks/challenges/series/SERIES.md` were read without modifying or executing their integrations. Scientific promotion still requires the series' matched conditions, isolated scoring, cost accounting and frozen artifacts.

## Missing connection

The plugin sends typed lifecycle operations to `drift.runtime.service`. Its current setup path turns UTF-8 bytes into at most 64 local token IDs; it is a reference setup path, not the GLM/Qwen tokenizer and chat-template path. `drift.runtime.builders.load_adapter` has Torch registry loaders for `qwen4_exp/torch` and `glm5_next/torch`, with no oMLX Studio or live GLM worker loader. The real `scripts/live/drift_loop.py` coordinates a different live path and is not controlled by this service protocol.

A successful SETUP/START response therefore cannot certify that the OMP agents use the evaluated GLM/Qwen memory pipeline. It also does not certify tool execution, remote cancellation, memory release, source ownership or completion of a repository task. The service currently has no authenticated live-backend capability negotiation.

## Implemented prerequisite

`/drift start` now refuses unsupported live startup before opening a socket or sending staged task text. `/drift start --reference <manifest-path>` explicitly opts into the existing reference service; the board, startup message and model-turn status identify reference mode and blocked live integration. This flag is an explicit operating-mode acknowledgement, not attestation of the server implementation or scientific readiness; it must not be advertised as enabling the real pair.

Command dispatch, startup parsing, connection settings, request validation, active operations and lifecycle cleanup now have separate modules. The old command module remains the command-registration entry point and re-exports `connectionSettings` for compatibility. Behavioral tests are split into startup, lifecycle and connection suites with a shared test harness. Multiline source comments in the plugin were reduced to single-line comments; extended rationale belongs here and in the protocol documentation.

The regression first failed because an unqualified live request connected and started the fake service. It now checks zero connections, zero requests and no displayed run for that request, while the existing reference lifecycle still passes. Parser tests reject a reference flag hidden in task text and preserve setup/checkpoint parsing for explicit reference mode. None of these synthetic tests demonstrates live repository collaboration.

## Next implementation gates

1. The installed provider/tool-call contract now passes the no-inference probe in `OMP_PROVIDER_CONTRACT.md`; build a worker-session adapter around the real tokenizer, chat template, request IDs and persistent native state, since ordinary completion responses alone do not expose the required cache operations.
2. Add authenticated capability and manifest binding before any real-worker start, including frozen model and translator hashes, supported lifecycle operations and approved per-session limits; reject reference, absent or unsupported capabilities for a live request.
3. Qualify acknowledged pause, cancellation and remote memory release, plus the causal delivery and source-ownership prerequisites, on the actual workers under approved limits; a killed SSH client is insufficient evidence that server-side inference stopped.
4. Connect the qualified session adapter to OMP tool dispatch with private worker identities and declared repository/artifact access, then evaluate a bounded repository task against Duo with external hidden checks and measured cost.

No new model load, inference, training, remote mutation, MCDMA startup, installation or publication occurred for this change. GPU time, token costs and energy were not measured; the work used local TypeScript tests only. The initial plugin baseline was 26 passing tests; the failing regression was observed before the fix; the final plugin suite has 29 passing tests and TypeScript checking passes.
