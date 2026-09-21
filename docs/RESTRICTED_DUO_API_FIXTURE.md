# Restricted two-agent public API development fixture

This is an auxiliary development fixture using unchanged stock Duo, two private worker sessions and deliberately restricted tools. It is not the ordinary full-tool Duo baseline, an activation-only test, a hidden evaluation, or evidence that native GLM/Qwen have completed a project together. Both agents receive the same public task file tools: read api.py/verify.py, replace the complete api.py and run the fixed public verifier. Parent additionally retains task/hub/todo; child retains hub/yield. The derivative guard checks advertised tools and every execution, exact model/cwd identity, one pinned DuoPeer dispatch and two-party Hub addressing, while rejecting inherited read/bash/edit/write tools, recursive children and paths outside the public API files.

The fixture reuses the existing API handler and public HTTP checks: GET /health must return HTTP 200 with JSON status ok, and a missing path must retain JSON 404. The whole-file operation is explicitly called drift_task_write and tells the model to supply every retained line. The API task directory contains no owner manifests, provider sources, receipts or hidden data; scope configuration and controller code remain outside the editable task root. The verifier script is pinned and denied writes inside its OS sandbox. Successful public verification records the API hash, and the final controller requires that hash to equal the final artifact; edits after a successful check cannot inherit that pass.

A concrete existing API-tool limitation was reproduced without changing the original tool: network_loopback=true permits all local ports. The development-only verifier instead permits its one owner-selected localhost port and denies all other networking, process fork and sibling/private filesystem reads; only bounded runtime roots and the assigned task directory are available. Regression checks demonstrate the previous broad loopback policy permits another port while the new policy denies it and a sibling private file. The owner must select a port reserved for this public fixture, and the program fails if it cannot bind it; this does not grant a general browser, shell or local-service tool. The controller itself is trusted owner code, not a same-UID sandbox; model-authored Python executes only in the verifier subprocess sandbox.

## Executable qualification

The actual installed OMP executable, unchanged stock Duo and actual Python WorkerSession core ran with synthetic workers against the real file/HTTP tools: parent six streams, child five, one task, one todo, three Hub calls, one successful public verification, both sessions shut down, no guard blocks or worker errors. Final public API SHA256 was `d39e0ff9b347fcf14cae9ff697d4d8a85d47795b215d765a262cdc3f16c68c12`. The child used the runtime-advertised yield schema, which in this installed executable uses top-level data; adjacent source described result.data and initially caused rejected fixture yields. The fixture now derives that shape from the actual advertised schema. Synthetic token counters were parent 18 input/12 output, child 15 input/10 output; they are not native inference costs or proof of independent reasoning.

```sh
python3 scripts/omp/development_fixture.py --omp /home/example/.local/bin/omp --source /path/to/integrated/tree --duo /home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo --port 49273
```

The synthetic controller is not wrapped in the earlier broad sandbox because macOS rejected nested sandbox_apply; the actual model-authored API verification subprocess retains its tested OS sandbox. The synthetic fixture registers no external model endpoint and performs no model/host inference. An earlier batched multi-tool producer repaired the public artifact but then failed the provider context-history guard; it is not counted as a pass or used to relax that guard. The qualified producer emits one tool call per turn and explicitly yields after its Hub report.

## Native preparation and limits

```sh
python3 scripts/omp/development_api.py --profile /private/owner/room-profile.json --profile-sha256 PROFILE_SHA256 --evidence /private/owner/fresh-development --port OWNER_SELECTED_PORT
python3 scripts/omp/development_api.py --profile /private/owner/room-profile.json --profile-sha256 PROFILE_SHA256 --evidence /private/owner/fresh-development --port OWNER_SELECTED_PORT --prepared-sha256 RETURNED_HASH --run
```

The same owner-reviewed native profile schema and freshness/cleanup readers from NATIVE_ROOM_QUALIFICATION.md are required; profiles remain explicitly no-link, multi-turn and limited to two pinned workers, six turns, 512 output tokens per turn, 20k cumulative accounted tokens and 50-second worker deadlines. The controller retains its existing independent shutdown bounds, hashes/discards generated stdout/stderr, requires both cleanup receipts and refuses replay of an evidence directory. Source/configuration pins and the fixed task prompt are reviewed before dispatch; no native dispatch was made for this ticket. GLM reconstructs and counts prefill each turn, so these unchanged room budgets may fail before a six-turn development sequence completes; raising them requires separate owner/resource review, not a silent fixture change.

The subsequent explicit reconstructed-worker allowance in NATIVE_ROOM_QUALIFICATION.md also applies here: a newly frozen GLM profile may declare 65,536 cumulative tokens while Qwen stays at 20,000, without changing turn/output/time limits. The controller validates outcomes against each frozen worker allowance; the same allowances must be used across any matched development arms. This does not alter the original failed room profile or establish that its unlogged second-turn rejection was caused by that limit.

A separately prepared `restricted-api-development` profile supports a 170-second worker/controller task deadline, ten turns and 180-second independent remote supervisors, with 174 seconds before local controller termination. The original room profile remains fixed at 50 seconds and six turns; its controller rejects the development purpose. The longer API profile retains 512 output tokens per turn, 262,144 input bytes and cumulative allowances of 65,536 for GLM and 20,000 for Qwen. These are profile limits, not authorization to run or claims of successful qualification. Historical attempts and limitations are archived privately. The same frozen allowances and isolated compaction setting must apply to both future matched development arms.

After the real API artifact passed but GLM's repeated full-prompt accounting cut
off final coordination, the owner questioned that artificial allowance. A fresh
development profile can now declare 168,960 cumulative GLM tokens, sized as ten
declared 16,384-token context windows plus 512 output tokens per turn. This is an
accounting allowance within the already authorized three-minute/ten-turn run,
not a hardware requirement, a larger model context, or permission for a longer
run. Qwen retains 20,000 incremental tokens; its measured usage already fits.
Default room profiles and every failed native profile remain unchanged, and any
future matched arms must share the newly frozen limits. Input bytes, output per
turn, model turns, wall deadlines, memory guards and cancellation remain bounded.

## Matched comparison and the remaining activation seam

The current runnable arm is explicitly text_only and declares peer-text plus shared-public-artifact channels. activation_supplement fails closed before loading a profile or dispatching a worker. A valid auxiliary comparison must retain this identical task, public verifier, tool permissions, both declared channels and resource limits in both arms, differing only in the separately pinned activation intervention; it must not be described as the unweakened ordinary Duo primary benchmark.

The exact existing bridge blocker is scripts/omp/experimental_extension.mjs: allowTextDuo and observed execution receipts are enabled only when experimental_multi_turn and memory_mode=no-link. Linked sessions therefore reject stock Duo developer/Hub text. The minimum future provider change is an explicit owner-pinned declared-text/artifact communication mode accepted identically in both auxiliary arms, while strict activation-only mode remains unchanged; the candidate additionally needs the qualified owner broker/outbox/translator/snapshot/append route and measured costs. This fixture does not implement that provider authorization or activate a link. Ordinary Duo source and baseline permissions remain untouched.

Local validation at base `d8756bb`: two focused Python preparation/channel tests passed, and two focused Bun policy/OS-sandbox tests passed with 15 assertions; the OS regression also reproduces the broad-loopback behavior of the unchanged previous runner. The full reference suite passed 705 with three existing adapter skips in 38.48 seconds; the selected `.venv-next` adapter/runtime suite passed 155 with one existing sparse-indexer parity xfail in 8.20 seconds; full plugin passed 84 tests/324 assertions and TypeScript passed. Commands were the established primary `.venv`/`.venv-next` Python executables with `PYTHONPATH=.` in this isolated worktree, `bun test plugin/omp-drift/test`, and the primary dependency TypeScript executable with this worktree tsconfig. No native model cost, money or energy was measured.

SHA256: development guard `2e249dbfdf223a3e6e4e2b3ea003ddd931f66c2a590c01b8991b37164dec3876`, port-scoped verifier `e68f771622dca3455f0bd2f5d7bc4c30d558b4bedc77ee600cde285f42361d66`, native preparation/controller `b68e41c07b61d3bc033e7864f9347984c1aaf030f115f20f98e55e9c59ec24d3`, actual-executable synthetic probe `807a325e7e85df7d2adf3a3600b8c944d4861af1d60d83a00a9c615666858215`. Next ticket: parent review of native room outcome/resource headroom, then fresh no-link development profile preparation; activation supplement remains BLOCKED pending its explicit declared-channel provider mode and independently qualified activation coordinator.
