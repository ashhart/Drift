# Current status

As of 22 September 2026. This is the canonical current-status summary; dated
research and engineering records describe the particular runs they document.
Documentation cleanup does not rerun an experiment or change its verdict.

## What works

| Path | Evidence | Scope |
| --- | --- | --- |
| Local reference and tap CLI | CPU regression suite and fail-closed project validation | Inventory, adapter scaffolding and existing translator validation; no automatic translator training |
| Native GLM/Qwen MCDMA exchange | Three confirmed two-way epochs through a controlled installed-OMP parent/subagent workflow | One tested pair and constrained lifecycle, not arbitrary stock Duo tasks |
| Linked cancellation | RPC abort after one confirmed epoch, with the next epoch staged | Both routes failed closed, neither next tool ran and both workers were independently reaped |
| Reciprocal recall diagnostic | One development fact per model | Engineering evidence, not a held-out reliability score |

The native lifecycle run took 30.034 seconds including startup and generation;
the separate cancellation run took 22.038 seconds in total. Neither is a
per-epoch transport latency measurement. Exact scope and receipt hashes are in
the [native guide](guides/NATIVE_OWNER_EXCHANGE.md) and
[dated evidence record](history/engineering-2026-09-22.md#2026-09-22--native-three-epoch-lifecycle-and-linked-cancellation).

MCDMA carries the designated activation payload, not SSH. Setup prompts, tool
results and counted task-control text still exist outside that channel; the
results do not prove that the whole workflow communicates only through KV.
Process reaping is not a general GPU allocator-release guarantee.

## What remains unqualified

- Automatic native startup from `/drift`, stock Duo room integration and arbitrary nested subagent workflows.
- General two-way recall reliability, source attribution and long-context quality.
- Complete final-tail coverage and repeated-run reliability; an earlier control closure remains unattributed.
- Portable two-host tap manifests and other model families without their adapters, trained translators and qualification evidence.
- The formal M-1 through M6 research gates merely on the strength of these engineering runs.

The shell CLI creates tap projects using supplied translators; it does not train
new translator weights. `/drift` currently controls the reference service and
does not convert an existing Duo room into a native KV-linked pair.

## How to interpret evidence

Transport ACKs, cache-application receipts and correct answers are different
checks. Keep failed and invalid attempts, no-link controls and costs visible.
Required environment skips remain BLOCKED adapter gates, not passes.

Historical lookup experiments and their controls are retained in
[miss diagnosis](evaluation/MISS_DIAGNOSIS.md) and the
[historical release assessment](history/RELEASE_READINESS.md); the formal Test A
minimum-improvement criterion failed despite a high aggregate accuracy.
Those results must not be attributed to a different reader recipe or schedule.
See [evaluation guidance](evaluation/README.md) before reporting a benchmark.
