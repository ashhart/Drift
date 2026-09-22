# Installed OMP room compatibility fixes

At base `8a1f73f37e2ca4a8adea3b431e54ccebf994631d`, the second native room failure was reproduced without model inference using the installed OMP executable and unchanged stock Duo. This is an engineering compatibility check, not evidence that real Duo collaboration or the API task has succeeded.

## Concrete defects and corrections

The installed Hub schema supports `list`, bare `wait`, and `wait` with `from`; the room guard rejected these legitimate forms. The bounded guard now accepts live discovery with at most two results and normalizes every accepted wait to the one pinned peer and at most 2,000 milliseconds; zero previously meant indefinite waiting in OMP, so it is replaced with the finite bound. Process/job controls, unrelated targets, persisted/parked roster requests and larger limits remain rejected. The same normalization is applied by the derivative development guard because it shares the room policy. Stock Duo and ordinary baseline tools are unchanged.

The registry is process-local, not inherently room-private: this allowance relies on the dedicated fresh OMP process/configuration, no restored session, and exactly one allowed child enforced by the wrapper. The executable fixture records only the result count and whether every returned identifier is Main or DuoPeer; these runs returned an empty roster after the peer had moved on, so they do not prove roster isolation in a shared OMP process. Running this qualification guard in an existing multi-room process is unsupported.

The provider CONTEXT defect was independently reproduced with a first assistant turn containing both todo and task calls. OMP stripped todo's intent argument in the observed execution while retaining the original argument in history; Duo also amended task arguments. Reconciliation blindly applied both receipts when only the task history had changed, thereby rewriting the unchanged todo sibling and failing its own equality check. Reconciliation now leaves already-equal parts intact and applies exact observed arguments only to differing parts; it still requires the complete resulting assistant message to match. Original generated assistant history remains unchanged, and both execution amendments remain explicit own-control updates before tool results. Unobserved arguments, identity/order/content changes, and forged receipts still fail closed.

## Reproduction and verification

The focused test failed before the receipt change with DRIFT_WORKER_CONTEXT. The actual OMP multicall fixture also failed at context:CONTEXT before any result reached its worker, then passed after the change. The fixture preparation modifies only copied public deterministic fixture sources, repins copied backend hashes before launch, and preserves stock Duo. No private native input or output was inspected for this diagnosis.

```sh
python3 scripts/omp/room_compat_probe.py --omp /home/example/.local/bin/omp --worker-source . --duo /home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo --multicall
python3 scripts/omp/room_compat_probe.py --omp /home/example/.local/bin/omp --worker-source . --duo /home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo
python3 scripts/omp/development_fixture.py --omp /home/example/.local/bin/omp --source . --duo /home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo --port 49273
```

Actual installed executable SHA256 `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a`; stock Duo command source SHA256 `ca699d1e07b273da2d95dac696c97d71d05aaa08260ea5fba91782097cf89e21`. Multicall: PASSED, parent five/child five turns, one open/close each, one todo/task and three Hub calls, zero errors, 1.238 seconds. Hub forms: PASSED, parent six/child five turns, four Hub calls, zero errors, 1.017 seconds. The existing restricted API executable fixture also PASSED with parent six/child five, one HTTP public verification, zero blocks/errors, both shutdown, final API SHA256 `d39e0ff9b347fcf14cae9ff697d4d8a85d47795b215d765a262cdc3f16c68c12`; synthetic counts were parent 18 input/12 output and child 15 input/10 output. These are fixture counts, not inference costs; no native money/energy measurements were made.

Focused Bun tests: 14 passed/92 assertions. Full Python reference: 713 passed, three existing adapter skips, 38.22 seconds; selected `.venv-next` adapter/runtime suite: 155 passed, one existing sparse-indexer parity xfail, 7.72 seconds. Full plugin and TypeScript checks passed. Commands used the main checkout's existing `.venv` and `.venv-next` interpreters with this worktree `PYTHONPATH=.`, `bun test plugin/omp-drift/test`, and the main dependency TypeScript executable with this worktree tsconfig. Next gate is parent review and a newly pinned bounded native room/API attempt after real-tokenizer input preflight; failed native evidence and all frozen profiles remain untouched.

Final plugin result: 92 passed/382 assertions in 1.68 seconds. SHA256: receipt reconciler `7975e2bc229945a35ed59843e94c1313f3a7cc5a7b827ab13abb369cea1b992f`; bounded Hub policy `0f8a10bc880e9a5fe66b601f4fcf947d2da644102f274f0322ffadf4668ac55a`; executable compatibility probe `7b1784c4d199acc5fda5ba1ab5bf29b90418ec278dd3b738cb17a310f104b7b9`.
