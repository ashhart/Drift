# Two-agent repository workflow release

> Historical record, preserved from the pre-cleanup documentation.
> Not current operating instructions; see [current status](../STATUS.md).

Historical host observations below are not a current deployment inventory. Public
examples use anonymized endpoint names; verify private manifests and deployed hashes
before any native run. The composed exchange defects were repaired locally in
`4dee39b`, but no new native recall or Duo qualification follows from that repair.

Owner target: two OMP agents working on one repository, using GLM-5.3 and
Qwen3.8-Flash, with shared memory and correct source ownership.

Current release verdict: **BLOCKED** for that complete workflow.
The lookup result and the local engineering checks below do not establish it.

## Intended integration

```text
OMP user interface / omp-drift lifecycle controls
                  |
        authenticated worker service
                  |
     GLM on Sparks <-> Qwen on Studio
                  |
     isolated repository/tool execution
```

Duo is the existing text-based comparison workflow. `omp-drift` is a separate
OMP plugin, with an opt-in worker provider exercised through unchanged stock Duo
using both synthetic and native workers; the native API runs currently use text
and shared artifacts, not a live Duo-drift project integration. The model
workers own private caches and memory exchange; OMP owns task lifecycle and
presentation. The live coordinator currently uses SSH and activation files.
MCDMA is a later transport backend, not the agent runtime or a prerequisite
for the first qualified SSH-based workflow.

## What exists, and what remains disconnected

| Component | Current evidence | Release gap |
| --- | --- | --- |
| GLM -> Qwen lookup transfer | Test A: 120 passages / 240 questions, trained 97.92%, untrained 95%, text 99.17% under the fixed reference-containment rule | Does not establish tool use, source ownership or sustained collaboration; formal Test A verdict FAILED on its minimum improvement criterion |
| OMP plugin | Behavior/protocol suite and TypeScript checks pass; exact counts and source pins are in the progress record; live start guard retained | Full Duo-drift coordination remains unqualified |
| Installed OMP provider/tool interface | Both native two-turn echo probes pass; GLM restores foreign snapshots on both ranks each turn and Qwen retains its native cache across tools | Complete native Duo orchestration remains unqualified |
| Bounded bidirectional exchange | Both real models completed a tool round after one selected Qwen row became GLM bank v2 and eight GLM generated rows were appended to Qwen; four rank receipts and both clean supervisor exits, 19.363 seconds | One paused subset exchange only; semantic use, source ownership, complete own-input export and final-tail coverage remain unproven |
| Stock Duo orchestration | Installed OMP with unchanged Duo passed synthetic checks; latest native run reached both models and three Hub calls with no guard blocks or CONTEXT failure, but hit the frozen 50-second limit | Completed room/API task and matched comparison remain unqualified; Hub text is a declared communication channel |
| Reference worker service | Authenticated lifecycle, registry-based worker construction, tested scheduling | Existing builders load local Torch adapters; no ready remote oMLX/vLLM worker builder |
| Real-serving coordinator | Exploratory simultaneous generation; no-link, failure, repetition, bounded pending delivery and publication-validation regressions | No qualified prior-epoch schedule, two-worker coordinator cancellation or end-to-end repository workflow |
| Live receivers | All layers validated before writes, cursor commits after device completion, failed sessions poisoned; every configured rank needs matching admission and post-write receipts | Both real ranks acknowledged one identical twelve-row publication; receipts detect failure but do not make writes atomic or prove cache-content equality |
| Scoped cancellation | Real GLM no-link and one-publication linked probes PASSED; Qwen source capture and native echo children exited and reaped normally, including failed attempts | Simultaneous Qwen/coordinator cancellation and allocator release remain unqualified |
| Translator recipe | Explicit hash-pinned base/fan-out/v4 selection; source cursors and expanded reader-row budgets tested separately | `drift_loop.py` still defaults to the older `stacked2` readers; no continuous recipe is qualified and the 97.92% lookup result must not be attributed to this live loop |
| Source ownership | Existing split-knowledge observations and failed concise-prompt intervention | No demonstrated fix on the real GLM/Qwen pair |
| Repository tools | Both real GLM and Qwen produced the isolated HTTP handler; public checks passed in attempts five and six, but both whole sessions FAILED, most recently with protocol/turn errors | No project activation link, hidden evaluator or quality comparison; peer-private separation and conflict handling remain unqualified |
| MCDMA | Native integration reference and separate research | Drift's MCDMA transport remains a deliberate qualification guard; no daemon startup authorized here |

## Release sequence

1. Finish bounded delivery, receiver validation, cancellation and causal scheduling
   in the real-serving path, with regressions before fixes.
2. Freeze a small development smoke-test manifest with explicit token, time and
   memory limits, then qualify the exact workers on the available hosts.
3. Demonstrate source ownership on development cases without tuning on Test A's
   held-out items, and preserve unsuccessful runs.
4. Connect the qualified worker lifecycle to `omp-drift`; retain Duo as the text
   arm and declare repository artifacts as a separate communication channel.
5. Run an isolated two-agent repository task with fixed budgets, a no-link arm,
   Duo baseline, externally protected checks, measured costs and failure handling.

No timer or shipping deadline turns an unmet gate into a pass. A private
experimental demo can be labelled as such, but the requested reliable workflow
requires the evidence above. Native GLM tool use, one guarded Qwen source capture,
and one bounded bidirectional native tool round have now run; no training, serving
restart, MCDMA daemon or public upload was performed by this qualification work.

Latest completed local gates also include bounded pending delivery, immediate
local process interruption and receiver admission/poisoning; exact commands,
commits and hashes are recorded in `docs/agent-progress.md`.

The review follow-up adds safe numeric error diagnostics and sanitized stack locations,
all-host failure polling, rank-specific receipts and terminal rejection of undelivered
publications. The five-module Spark bundle is now deployed on both ranks, with both
staged imports and post-restart active-inventory checks PASSED at receipt commit
`a1d4eea`. The current read-only recheck also passed, with health 200 and zero
running/waiting requests and KV-cache occupancy. Deployment and one standalone
GLM cancellation probes and one linked rank write are now PASSED; remaining
workflow gates are described in `docs/reference/exchange/TRANSLATION_BRIDGE.md`.

The original bidirectional-round and failed native Duo room receipts are archived
outside the repository; see `docs/reference/omp/LINKED_NATIVE_ROUND.md` and
`docs/reference/omp/NATIVE_ROOM_QUALIFICATION.md` for the current protocol requirements.
The separate fifth and sixth native development runs produced an API passing its
public checks, but failed whole-session completion, as recorded in
the privately retained development receipts; this does not establish the paired comparison.

The owner has requested Duo-drift versus normal Duo on a project after these
gates, comparing completion and code quality under matched conditions; the
comparison preparation does not authorize bypassing the remaining gates.
