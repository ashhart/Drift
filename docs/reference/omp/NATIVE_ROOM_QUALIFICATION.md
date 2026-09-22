# Native room-only qualification controller

This prepares a bounded, explicit no-link GLM/Qwen room controller around unchanged stock Duo and the separately tested room guard. It does not qualify the ordinary full-tool Duo baseline, project development, activation-only communication, or semantic usefulness of exchanged messages. No real worker or host was dispatched while implementing it. The actual installed OMP executable ran the new extension against the actual synthetic WorkerSession backends: parent six turns, child five, one task, one todo, three Hub executions, both session shutdowns, zero guard blocks and zero errors. Synthetic usage is not a model cost estimate.

## Frozen profile

The owner JSON continues to use the existing experimental provider schema and command/artifact pin validation. It must contain exactly two distinct live workers, one unique session per worker, `memory_mode: "no-link"`, `experimental_multi_turn: true`, explicit retained/reconstructed state and exact model/translator hashes. The room accepts only model IDs made from letters, digits, `_`, `.` and `-`, up to 96 characters. Only the two selected provider/model routes can run; the guard checks their actual context and exact scratch cwd, permits one pinned DuoPeer task and removes/rejects inherited file/shell/edit/recursive tools. Peer Hub/developer text remains declared text communication on private own-input channels, never activation transport.

Both owner and actual backend configurations must use these exact limits: `max_input_bytes: 262144`, `max_output_tokens: 512`, `max_session_tokens: 20000`, `max_turns: 6`, `deadline_ms: 50000`. The 20k cap is cumulative accounted input plus output, including native reconstructed prefill on each GLM turn and retained input deltas on Qwen, not a 20k context-length assumption; startup costs outside terminal accounting remain separate. OMP max-time is 50 seconds; the local controller waits at most 54 seconds and reserves up to six seconds for its process/output cleanup, within the existing owner launcher 60-second ceiling. Independently pinned remote supervisors must retain their own cleanup envelope. Cleanup readers have a five-second bound each; those read-only receipt checks occur after controller shutdown and do not grant more inference time.

Development amendment after the first failed attempt, before another native dispatch: an explicitly pinned reconstructed worker may instead use `max_session_tokens: 65536`; retained workers remain at20,000 and all turn/output/time limits remain unchanged. The first GLM generation alone counted9,212 prompt tokens, which must be charged again when its transcript is reconstructed; a20,000 cumulative allowance is not a20,000-token context window. The larger allowance is a separate fresh engineering profile, not a retrospective change to the failed run or a proved diagnosis of its unrecorded second token count. The report now checks each worker against its own frozen allowance. Any later matched project comparison must freeze these same worker-specific policies across all arms.

The profile is a bounded JSON file (at most 65,536 bytes) with exactly these fields:

```json
{
  "version": 1,
  "purpose": "restricted-no-link-room",
  "owner": {"path": "/absolute/owner.json", "sha256": "OWNER_SHA256"},
  "omp": {"path": "/absolute/omp", "sha256": "EXECUTABLE_SHA256"},
  "duo": {"path": "/absolute/stock-duo", "sha256": "SOURCE_TREE_SHA256"},
  "parent": "drift-experimental/OWNER_PARENT_MODEL_ID",
  "child": "drift-experimental/OWNER_CHILD_MODEL_ID",
  "cleanup": [
    {"worker": "OWNER_PARENT_WORKER", "session": "OWNER_PARENT_SESSION", "command": "PINNED_READER_COMMAND_OBJECT"},
    {"worker": "OWNER_CHILD_WORKER", "session": "OWNER_CHILD_SESSION", "command": "PINNED_READER_COMMAND_OBJECT"}
  ]
}
```

The uppercase values are intentionally invalid placeholders, not usable host endpoints. Each cleanup command object uses the existing command shape `{executable, sha256, args, cwd, env}` plus a nonempty `artifacts: [{path, sha256}]` list covering its reader code/configuration; worker commands and cleanup commands are exact owner inputs, with no shell interpolation or inherited environment. Accepted environment keys are PATH, HOME, SSH_AUTH_SOCK, PYTHONPATH, PYTHONDONTWRITEBYTECODE, OMP_NUM_THREADS and OPENBLAS_NUM_THREADS; no environment is inferred or copied from the controller. The owner is responsible for pinning its verified SSH profile, host-key policy and existing agent socket when needed. The wrapper never installs keys or weakens host verification.

Each cleanup reader must refer to that worker's fresh, owner-bound supervisor receipt: before dispatch it must return exit 3 with empty stdout/stderr because the receipt does not yet exist; afterward it must return exit 0 and one JSON aggregate receipt, at most 8192 bytes. This checks freshness without inventing a remote API or printing remote paths/credentials. The same reader cannot be assigned to both workers. Direct `worker_terminated` receipts and existing owner-stdio receipts containing `worker` are accepted only with exit 0, reaped child, no remaining process group, no TERM/KILL escalation, allowed normal termination reason and at most 60 seconds; owner-stdio additionally requires PASSED, confirmed cleanup and joined owner thread. Missing/old/incomplete receipts prevent PASSED, even if OMP reports success. Numeric cleanup aggregates and a receipt hash are retained; unknown text fields are excluded from the report.

Compute the stock source pin using `native_room_profile.tree_digest(Path(stock_duo))`, which hashes a canonical list of all source files plus the duo-peer agent. Profile loading validates local executable, owner, artifacts, exact selectors, native states, limits and source pins before a worker command can run. The prepared bundle pins its copied provider, guard, stock Duo source and scope, and uses a fixed user prompt shorter than 1024 bytes; arbitrary tasks and prompts are not accepted.

## Prepare and review, then explicitly run

```sh
python3 scripts/omp/native_room.py --profile /private/owner/room-profile.json --profile-sha256 PROFILE_SHA256 --evidence /private/owner/fresh-room
python3 scripts/omp/native_room.py --profile /private/owner/room-profile.json --profile-sha256 PROFILE_SHA256 --evidence /private/owner/fresh-room --prepared-sha256 RETURNED_PREPARED_SHA256 --run
```

Preparation is the default and runs no controller, cleanup reader, remote command or model. The evidence directory must be fresh and outside Git repositories; it contains a private scratch bundle, prepared command/environment/pins and later aggregate facts/report. Review the generated `prepared.json`, owner profiles, resource authority and returned SHA before `--run`. Dispatch revalidates profile/source pins and prepared hash, verifies that both receipts are absent, then exclusively creates `run.started`; the same directory cannot be used for another attempt. Generated controller stdout/stderr are hashed and discarded, with a 2 MiB cap per stream; timeout or surviving local descendants causes cleanup and failure. The final report requires exactly the two model routes, one task, at least one todo and Hub execution, bounded positive usage, both session shutdowns and both independent cleanup receipts; it makes no claim about message correctness.

The native room controller deliberately does not grant model file or shell tools, and does not claim an OS network allowlist or a same-UID filesystem sandbox around owner processes. It uses fixed owner commands and the enforced OMP room tool boundary; widening this into project work would require separate tool-process filesystem/network containment and matching the ordinary Duo baseline, not merely removing the guard. No full project task, hidden test, peer-private file or raw activation is staged. Exact native commands, identities, source/owner pins and resource permission must be supplied and reviewed by the owner before the first real run.

## Local validation and handoff

Base: `ff3b2aa29a050f006dd9c55e43581f2a0532aed4`; initial focused regressions failed because the profile/evidence modules did not exist, then seven focused tests passed in 0.14 seconds, covering pins/caps, prepare-without-dispatch, stale/missing cleanup, private-text exclusion and process descendants. Commands were `PYTHONPATH=. /opt/drift/.venv/bin/python -m pytest -q tests/test_native_room.py` and the same command without the test path for the full suite: 628 passed, three existing adapter skips, 35.92 seconds. The established `.venv-next` selected adapter/runtime suite passed 155 with one existing sparse-indexer parity xfail in 7.77 seconds; `bun test plugin/omp-drift/test` passed 62 with 269 assertions, and TypeScript passed. No real model cost, money or energy was measured; synthetic token counters in the executable fixture were parent 18 input/12 output and child 15 input/10 output.

SHA256: controller `cc8f0c6db16f86f0c98bf1348527405b72c1a596059b166479d77b9e3b7afe0a`, profile validator `71b72de04c0b341a5f8be0e43105982d42fad1be8a52b87a34c01c7c022dcab9`, native room extension `b5c0d2dd1b4d72da9aed532877e6e4ee84e43f4b70e4af7992fb3c257f1eb906`, receipt validation `6cbea6b1063460b81cb935997397fa352c18628ede42f07a2c7801a1c3686cc0`. The next ticket is an owner-reviewed native profile with fresh exact worker commands/session/artifact pins and compatible fresh receipt readers, followed by preparation and only then an authorized bounded dispatch. Full-project tools and activation-linked room communication remain separate, unqualified stages.
