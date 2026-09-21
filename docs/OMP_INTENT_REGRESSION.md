# Installed OMP intent-metadata regression

Engineering verdict: PASSED for the reproduced provider defect; the first native GLM failure remains preserved and no native rerun occurred in this ticket.

The provider fingerprinted entire assistant content objects. Installed OMP adds controller-owned toolCall.intent after reading the model's required argument i; the next provider call therefore rejected the unchanged semantic answer as modified history. Adjacent source locator: `/home/example/.bun/install/global/node_modules/@oh-my-pi/pi-agent-core/src/agent-loop.ts`, extractIntent at 2176 and assignment to toolCall.intent at 2179. The exact installed executable, not that source alone, reproduced the defect.

Changing the fixture tool arguments from {} to {i: 'fixture intent'} produced two failing selector runs through actual OMP and actual Python WorkerSession: each had one successful real OMP tool call, one stream, zero submitted tool results, zero second streams and exit 1. That matches the location of the native failure, but does not prove the historical native exception without additional evidence. Standalone space/newline text fragments had already passed and were not the reproduced cause.

`worker_semantics.ts` now fingerprints generated text/type and toolCall type/id/name/complete arguments. It excludes only the verified controller-owned intent field and undefined optional properties. Original arguments.i is retained; modified i, other arguments, tool name, tool ID, changed text and unknown content fields are still rejected. The regression fixture permanently emits an intent argument plus multiple text fragments so future installed-runtime checks exercise this path.

Command: `python3 scripts/omp/registration_probe.py --omp /home/example/.local/bin/omp --worker-source /home/example/.codex/worktrees/drift-worker-session/Telepathy`.

After the fix both selectors passed: open 1, own_prompt 1, tool_result 1, stream 2, close 1, tool_calls 1 and other_tools 0. Wall time 1.151 seconds; no model inference, monetary cost or energy measured. Executable SHA256 `d498da40d577e1ffa681ca8632c2ea40a9f722a08b880412011d37dffee9513a`; actual worker core SHA256 `91306a0d708a7f072ba53ae44499f2db87564f67bffb0ede5e9654645084ce8c`; stdio SHA256 `8f3cc565ec2bc81aff511e28c7986b3314889ed83746f2dcc2e2d554093e9bc1`.

Focused semantic tests: 3 passed after a recorded red metadata case. Plugin suite: 54 passed; TypeScript clean. Full reference: 407 passed/3 skipped; next: 155 passed/1 pre-existing expected failure. Next: parent integrates this fix, preserves earlier native reports, and decides whether a newly identified bounded native probe is authorized; no existing owner/session should be silently reused.
