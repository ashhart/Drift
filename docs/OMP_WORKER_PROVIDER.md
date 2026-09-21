# Private worker provider prerequisite

Engineering verdict: PASSED for a synthetic worker through the actual installed OMP executable; BLOCKED for a real GLM/Qwen coding workflow until the backend and resource gates are qualified.

The new modules are opt-in building blocks, not an automatically registered live provider, and the existing `/drift` reference-only guard remains unchanged. The private worker subprocess channel carries only that worker's own system prompt, tools, user inputs and tool results; it is separate from the activation transport. Its caller must supply a pinned worker executable, restricted environment, work directory, identity, model and translator hashes, limits and accepted backend/native-state capabilities. A hash echo binds a claimed identity and does not establish artifact integrity by itself; the launcher/backend must verify actual artifacts before claiming live qualification.

## Bound protocol

Every NDJSON envelope has exactly `v:1, session, worker, seq, op, payload`; request sequence increases and each reply echoes the requesting identity and sequence. `open` binds model_id/model_sha256/translator_sha256 and the five limits max_input_bytes/max_output_tokens/max_session_tokens/max_turns/deadline_ms, plus system_prompt string array and tool schemas. `opened` echoes all pins and limits with backend fixture/live and capabilities native_state retained/reconstructed/fixture, tool_calls and cancellation. Explicitly qualified reconstructed state is supported, and every terminal's input plus output tokens is counted, including repeated prefill.

`own_prompt{text}` gets own_prompt_ack; `tool_result{call_id,text,is_error}` gets tool_result_ack{call_id}; `stream{max_tokens}` yields text{text}, complete tool_call{call_id,name,arguments}, then terminal{reason:stop/tool_use/length,usage:{input_tokens,output_tokens}}; cancel{target_seq} gets cancelled{target_seq}; close gets closed. Errors, changed bindings, stale replies, limits or missing acknowledgements poison and close the client. Tool calls must name a registered tool, have unique IDs, and receive at most one result before another stream. Prefix comparison rejects changes to earlier own context instead of silently resending it; assistant content already generated is never replayed to the worker. Images and other unsupported content fail closed.

OMP mapping emits actual text, tool call, done and error events. Normalization removes undefined runtime metadata and sends only documented tool fields: the installed runtime exposes additional cyclic/private tool implementation state that cannot be sent as a schema. The one-shot closeOnStop option acknowledges worker close before the final OMP terminal: an observed CLI agent_end hook did not reliably finish asynchronous teardown before exit. Long-lived callers must explicitly close on session end and fail the session on cancellation; tool runtime cleanup and actual model cancellation still require backend proof.

## Observable qualification

Actual executable: `/home/example/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js`, OMP 18.2.6 compiled Bun executable, SHA256 `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a`.

Run `python3 scripts/omp/worker_probe.py --omp /home/example/.local/bin/omp`; it copies only provider/probe modules into a temporary directory, denies network with sandbox-exec, denies writes outside that directory and home reads except installed runtime dependencies/native module paths, and uses isolated OMP configuration/session state. It launches no model, uses no credentials and performs no real repository task. OMP actually executed the registered fixture tool and fed its result back through the provider to an owned Python worker: open 1, own_prompt 1, stream 2, tool_result 1, close 1; OMP tool_calls 1, tool_events 1, other_tools 0, closed true. The synthetic usage counter was 10; this is test accounting, not measured model tokens. Measured elapsed time 0.492 seconds; monetary cost and energy were not measured. Exit code 0; stderr empty.

Aggregate stdout SHA256 `16da0c8b6ef40f6b39d7a70b75cfc30456ec4856279d2e865b30970e2d193daa`; extension SHA256 `6dc63d5ab38eb42023c31413567366597ff1dca24e2d578fe779d45527b4a1fc`; sandbox policy SHA256 `6bc1e1f3d983374d907825bf117de70812895d97ea7ab0056f2c75cda06cd5b3` (temporary path makes policy/output hashes run-specific).

## Tests and handoff

Base commit `0c2822abfcfd86fa13b52644c2b5432538d69893`; changes are committed separately on codex/omp-worker-provider. Baseline plugin suite: 29 passed. Red evidence: missing worker_client module, then an explicit regression demonstrated extra opened fields were accepted; fixed with exact capability/manifest shape checks. Focused tests cover pin mismatch, unqualified/native reconstructed capability, per-turn prefill, duplicate tool results, acknowledged/missing cancellation, real owned child protocol round trip, changed context and terminal mapping.

Commands from this isolated worktree:

```sh
bun test plugin/omp-drift/test
/opt/drift/plugin/omp-drift/node_modules/.bin/tsc -p plugin/omp-drift/tsconfig.json --typeRoots /opt/drift/plugin/omp-drift/node_modules/@types
PYTHONPATH=. /opt/drift/.venv/bin/python -m pytest -q
PYTHONPATH=. /opt/drift/.venv-next/bin/python -m pytest -q tests/test_adapters_next.py tests/test_translate.py tests/test_core.py tests/test_transport_sync.py tests/test_runtime_train_mail.py tests/test_properties.py tests/test_wire2_pool.py tests/test_hive.py tests/test_mailbox.py tests/test_m1.py tests/test_service.py tests/test_workspace.py tests/test_e3_compete_registry.py tests/test_adapters_mlx.py
```

Results: plugin 40 passed, TypeScript clean; reference 407 passed, 3 skipped, 19.23 seconds; next 155 passed, 1 existing expected sparse-indexer-tie failure, 8.10 seconds. Skipped real adapter gates are BLOCKED, not passes. Neither fixtures nor reference suites demonstrate real native continuation, foreign-memory restoration, source ownership or repository collaboration.

Next ticket: register an explicit experimental provider against the separately qualified private worker backend, verify real artifact pins and cancellation/cleanup under approved limits, and run a bounded real tool round trip with counted prefill and verified foreign snapshot restoration. Preserve the reference guard until the capability evidence warrants changing it.

## Source hashes

| File | SHA256 |
| --- | --- |
| `plugin/omp-drift/src/worker_client.ts` | `4e557781da93ac2cc3062e2fccc585f71ae56ed30305d4327b0f743a7ca95404` |
| `plugin/omp-drift/src/worker_context.ts` | `8f29bffc19bf10b40f37b817fa3a4938d7245881bbc12169a1088415d8caf620` |
| `plugin/omp-drift/src/worker_ledger.ts` | `cd18999f71269d40d784f2fa9ea55e7ce0cbcd844c8e46adea1a0972b145c74d` |
| `plugin/omp-drift/src/worker_mapping.ts` | `ab4f93950fe843d32198f764bfeef5a2ebccd9e2b13d1622d389093e37476637` |
| `plugin/omp-drift/src/worker_process.ts` | `1b03ae2354f4d49da150b65d8ccf298b6769db6d437f539c91ca66a1ceaa8ea2` |
| `plugin/omp-drift/src/worker_protocol.ts` | `4c6c371a38ea9842793dabbc595e48e8a768182f571eb897223e312ae6629495` |
| `plugin/omp-drift/src/worker_provider.ts` | `9b02c8899e66d3229f941755192e7a1ac80718e816d7ad2f142fc573b4051789` |
| `scripts/omp/worker_contract_extension.mjs` | `6dc63d5ab38eb42023c31413567366597ff1dca24e2d578fe779d45527b4a1fc` |
| `scripts/omp/worker_fixture.py` | `a22fba5ea2e6e766428ae62f3c5c84d79cc2e07b0893a5adf52dbfeead810c41` |
| `scripts/omp/worker_probe.py` | `a6271805c0055c084cbe0c8a5ab5dd761b2388d2e7175156cfb2cbda4ab12eae` |
