# Duo text versus Duo-drift: bounded project comparison

Status: **BLOCKED — planning only**; no model run, hidden answer generation, scorer execution or resource reservation is authorized by this document.

The primary question is whether GLM-5.3 and Qwen3.8-Flash working as Duo-drift complete a project with better code quality and performance than **normal Duo with its existing text-based communication**. Normal Duo remains the actual stock workflow: equal peers, its existing Hub protocol, shared working directory, shared notes and todo behavior. Both members may share the declared project repository within one run; each arm starts in its own clean run clone, and evaluator data and runtime-private state remain outside both agents’ access. This is a practical project-work comparison informed by the M5/E4 artifact-accounting rules, not a claim of KV-only collaboration or a replacement for the stricter scientific stage gates. The owner selected a fresh isolated benchmark project; Stockledger is frozen at the commit recorded below, while resource approval remains pending.

## Evidence and prerequisites

The inspected Duo checkout is `/home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo` at `8aeb53d3377928874e364bc6f6f8a05b471983cc`, package `omp-local-duo` version `0.5.1`. Its `src/command.ts` selects the peer, starts `duo-peer` through a next-turn instruction and requests Hub cancellation; `src/prompt.ts` and `agents/duo-peer.md` define equal peers, negotiated file ownership, Hub questions and shared `.omp/duo/notes.md` notes. The peer has read/write/edit/bash/search/LSP/AST/Hub tools and cannot spawn further agents; the visible session records the canonical todo list. The baseline must retain this competent coordination policy rather than being reduced to occasional summaries.

Relevant local contracts are `TELEPATHY_AGENT_ENGINEERING.md` sections M2.4 and M5.2, `drift/eval/e3.py`, `drift/eval/workspace.py`, `docs/reference/omp/OMP_LIVE_INTEGRATION.md` and `docs/history/RELEASE_READINESS.md`; the reviewed Telepathy base is `9cad535084c6fdf5abecbb06d2a8ed9856463b84`, with integration work still uncommitted when this plan was written. Series rules were read from `/home/example/Documents/Projects/Personal/Vontra/benchmarks/challenges/series/SERIES.md`: strong baselines, matched conditions, isolated scoring, five trials per arm as a pilot floor, frozen manifests and uncertainty-aware verdicts apply. This plan is not the required frozen machine-readable evaluation manifest.

Before an evaluation may start:

1. Bind the actual OMP binary to inspected source and a tested provider/tool-call contract, then connect the real GLM/Qwen sessions to OMP; the current reference service is insufficient.
2. Bind authenticated capabilities to exact model, tokenizer, runtime, translator, gate and prompt hashes, including the selected live translator recipe; do not attribute Test A's v4 result to an older live reader.
3. Qualify acknowledged remote cancellation, memory release, deadlines, failed-rank propagation, bounded memory and prior-complete-epoch delivery on both real workers.
4. Demonstrate the required source-ownership behavior on separate development cases and qualify the pause/interrupt behavior used by the tool loop; retain failures rather than silently changing the claim.
5. Enforce separation between runs and between the declared shared repository and private runtime/evaluator state outside model instructions, then demonstrate leak rejection and complete channel/cost records with disposable fixtures; do not remove ordinary shared-repository access from the stock Duo baseline.
6. Freeze the comparison manifest and hidden-check ownership after development calibration; use the owner-selected frozen project and obtain resource approval before any model workload.

The E3 helper is a reference comparison utility, not a production budget-enforcing OMP runner. The workspace helper demonstrates a stricter private-checkout regime; it does not require changing the user’s requested normal-Duo primary baseline. E4 explicitly recognizes peer code, diffs and artifacts as a communication channel and limits claims accordingly. For this practical comparison, shared repository files and notes are declared artifacts, not peer-private memory; private runtime files, model state, prompts, evaluator records and other trials remain OS-isolated. Directory separation alone cannot enforce that boundary, and a shared Git store is permitted only inside the declared shared run, never as a path into another run or private state.

## Selected disposable project

The owner selected a fresh isolated benchmark project, now prepared as **Stockledger** at `/home/example/Documents/Projects/Personal/Drift-Benchmark`, commit `9059e94ad1fc47c29d25f29f9e568e7d588dd07b`, tree `f8e31939e427f2d01e9d4a63e2ef43f9f3c759ae`. It is a small standard-library Python warehouse-event CLI task: validation, persistent atomic/idempotent imports, filtered listing, deterministic daily reports and concurrent writes. `SPEC.md` contains the fixed public contract, modules are intentionally unimplemented, and four public smoke tests include three expected starting failures; see `docs/evaluation/DUO_DRIFT_PROJECT.md` for setup evidence and hashes.

The primary pilot uses five paired fresh development attempts on this same public task, each arm receiving the exact same pristine starter and brief. Every attempt implements the project afresh; repeatedly executing one completed artifact does not count as repeated development. Freeze any supported sampling seeds and order before launch, and verify whether actual worker randomness permits independent attempts; deterministic repeats must be labelled accordingly. These repetitions measure reliability and variability on this one public project, not five independent held-out tasks or generalization to other repositories.

No hidden checks, reference solution or answer store exists in the starter, and hidden evaluation ownership remains undecided. The task is visible to the engineering agents, so results must be described as project-specific performance. A later stranger test needs separately authored, unseen task variants after the runtime recipe freezes, its own budget, and an isolated independent scorer; none is being generated now. The user's live project working copy is not a trial target.

## Arms and matched access

| Condition | Peer coordination | Repository artifacts | Purpose |
| --- | --- | --- | --- |
| Normal Duo text | Unmodified inspected Duo behavior: Hub negotiation, questions, status, equal peers and canonical todo recording | Both members share the run’s repository, working directory and `.omp/duo/notes.md` under logged tool access | Primary stock text baseline |
| Duo-drift | The qualified Drift activation stream and frozen ownership/control policy; the selected product’s text fallback policy is recorded explicitly | Both members share their own run’s clean repository and have the same artifact permissions as normal Duo | Primary practical workflow comparison |
| Drift no-link | The same Drift setup and tool loop with the activation link disabled by a verified hard-off control | The same shared repository and notes permissions | Secondary, separately budgeted causal control |

The primary result compares the two practical workflows on completion, code quality and performance, including all their declared coordination and artifact paths. Normal Duo’s text communication is never disabled or weakened to make Drift look better. A no-link control is required only for the stronger causal claim that activation exchange itself produced a benefit, and its resource approval is separate from the primary comparison. Freeze whether the actual Duo-drift product replaces Hub messages or supplements them, count all text fallbacks, and label an additive product **Duo text plus Drift**; neither policy changes the stock-Duo baseline. Solo GLM and solo Qwen remain useful E3 controls outside this bounded project’s initial budget.

Each pair uses the same starting commit/tree hash, task brief, legitimate observations, public tests, tool versions, tool permissions, dependency cache, model checkpoints/quantization, sampling configuration, per-model budgets and allowed retry policy. The same model is the visible/recorder member in both arms of a pair; a later role swap is a separate balanced factor, not an unrecorded change. All setup prompts and any setup text exchanged between peers are captured and charged; shared task setup is not an unmetered communication exemption.

For each trial, create a separate pristine run clone from the same frozen starting commit, with no Git alternates or hardlinks into another arm’s objects. The two agents in that run share its project working tree exactly as normal Duo expects; the next arm receives a different clean clone and cannot inspect earlier artifacts, room history or notes. Capture shared-file reads, writes, diffs, command outputs and note updates through authoritative tool/OS auditing, recording bytes and producer/consumer identity where available. Peer code and notes are legitimate artifact communication in both arms, not a hidden channel or forbidden peer-private access.

Keep worker-private prompt/session/cache files, private scratch, raw activation files, credentials, evaluator records and audit logs outside the shared repository and outside the agents’ tool permissions. Enforce that with separate service accounts or equivalent process/filesystem boundaries, not instructions; permission to use shared project files does not permit reading another model’s process memory or private runtime state. Snapshot the final shared tree into an evaluator-only clone for hidden checks after both agents are stopped, with no scorer feedback returned to them.

Match read/write/edit, local shell, search, LSP and AST capabilities across arms; the visible member’s todo-recorder privilege remains identical. Preserve normal Duo’s existing file-claiming, negotiation and conflict-resolution behavior rather than inserting a new merge service or forcing per-worker branches into the primary study. The same outer host enforces the approved budgets and sandbox limits for both arms without changing Duo’s communication semantics. No extra model, delegation, internet, package installation, credentials, production services or privileged commands are available; dependencies and public tool documentation are prepared before the clock starts. Conflicting or overwritten edits and unresolved required work count against completion and quality, with no evaluator rescue.

### Optional stricter study

A **secondary controlled-channel study** may put each worker in a private clone and use a controlled artifact/merge service, following the stricter workspace helper and M5.2 regime. That requires explicit, disclosed adaptations to normal Duo’s shared-directory instructions and must be named **Duo under matched containment**. It does not replace the primary normal-Duo comparison, and its controls are required only for the narrower causal or channel-isolation claim being tested. A practical shared-repository result cannot by itself be used to pass that stricter scientific gate or to claim all collaboration occurred through activations.

## Proposed resource envelope and ordering

These are requested caps, not approved resources or predicted costs:

| Item | Proposal |
| --- | --- |
| Minimum primary sample | Five fresh paired development attempts on Stockledger × two arms = ten development runs |
| Per-run wall limit | 20 minutes, beginning before room admission/task setup and ending after the final artifact snapshot and verified stop |
| Generated tokens | At most 8,192 per model, 16,384 total, including reasoning, coordination and any local mailbox generation |
| Cumulative input tokens | At most 65,536 per model across all requests, including repeated context, prompts and tool outputs |
| Tool allowance | At most 128 invocations per pair, with each subprocess bounded to 60 seconds and the enclosing run deadline |
| Automatic retries | Zero whole-run retries; tool/request retries inside a run consume the same remaining caps and are logged |
| Primary maximum scheduled wall time | 200 minutes, excluding separately reported host preparation and evaluator time |
| No-link addition | Five more runs, up to 100 further scheduled minutes, only if approved |
| Host reservation | GLM on the two Sparks and Qwen on the Studio, one arm at a time; exact host-memory/concurrency limits remain unresolved |

The broker must enforce all ceilings, including stopping server-side work and acknowledging release; exceeding a declared cap is recorded, never rounded away. Exact tokenizer counts are separate for GLM and Qwen, so a token cap is not a claim of matched FLOPs or equal inference cost. Also report elapsed serving time and measured device use where available; missing energy, accelerator-time or monetary measurements stay null rather than zero. Prefill, peer-message rereads, projection, transport and extra decoding all count, even when a model reuses a cache.

Within each paired attempt, randomize which arm runs first using a frozen seed, keeping order approximately balanced across the five pairs; do not run arms simultaneously on the shared hosts. Start each arm with new native state, empty foreign banks, isolated room history, empty notes and a fresh shared run clone, then verify cleanup before the next arm. Record load/preflight conditions and actual host contention. If asymmetric cold starts cannot be eliminated, include them in the primary wall figure and report a warm-serving breakdown separately; do not selectively exclude slow setup.

## Hidden checks and code quality

An independent experimenter owns the authoritative hidden checks, feature weights and quality rubric outside every model-readable repository, mount and service account. Tests are materialized only in the evaluator's isolated copy after artifact hashes freeze, and scorer output is never returned to the working agents. The evaluator runs submitted code without network, credentials, access to other trials or writable scoring infrastructure; subprocess limits and read-only measurement records prevent submitted code from changing its own score.

Primary outcome: **complete within the declared budget**, requiring a runnable final repository artifact, all mandatory public and hidden acceptance checks, no forbidden action, no unresolved required feature or merge conflict, and verified termination. A model saying "done" is not completion. Record a feature-weighted acceptance fraction as a secondary outcome so partial work remains visible; freeze weights and required-check categories before the run.

Quality is measured separately from completion: externally verified regression count, interface/requirement conformance, type/lint/build failures, error-handling failures, and a blinded review of maintainability and test adequacy against a frozen rubric. Reviewers receive anonymized final diffs with arm labels removed; no self-review by the working model counts as authoritative quality. Do not substitute lines of code, test count, model confidence or a post-hoc blended score for quality. Preserve tests added by agents, but authoritative hidden checks do not trust modifications to public tests or build scripts.

Scope the evaluator to objective specification compliance; neither the runtime's own ownership labels nor a successful cache delivery earns project-quality points. Ownership mistakes, false completion claims, test tampering and broker violations receive explicit incident records regardless of the final pass fraction.

## Communication and overhead ledger

Record each direction separately and distinguish the live activation channel from every other permitted channel:

- Activation payload bytes, framing/metadata bytes, publications, acknowledged/failed deliveries, rows, projection time, transport time and peak queued/resident memory.
- Hub messages, exact transmitted text bytes and sender/receiver tokenizer counts, plus prompts or summaries that repeat them.
- Repository and note reads/writes, patch/merge bytes, paths, producer/consumer identities, hashes and timestamps, including commits and filenames used to communicate decisions.
- Public test/tool outputs delivered to either worker, canonical todo updates, room-status messages and any shared scratch or integration feedback.
- Own-context prefill/decode, hidden reasoning when metered by the runtime, local mailbox work, tool execution, idle/wait time, setup, failed requests, retries and cleanup.

Audit records and private activations are not shown to either model and are never published automatically. Shared notes and code already carry text, so a win supports this complete audited workflow, not a claim that the models exchanged no text at all. Report artifact traffic in both arms; the primary comparison can support a practical workflow advantage, while attributing that gain specifically to the activation link additionally requires the no-link control. Development/training/calibration cost is reported separately from these operating costs, with unmeasured historical costs marked unknown.

## Analysis, failures and verdict

Freeze a machine-readable manifest with the primary completion outcome, paired-attempt sampling unit, worthwhile improvement, quality/overhead constraints, trial count, order seed, cap enforcement, invalidation rules and uncertainty method. Proposed worthwhile improvement for an expanded study is a 10-percentage-point absolute completion gain with no increase in critical quality failures; the owner must accept or replace it before evaluation, and five pairs cannot reliably resolve such a small gain. The pilot reports feasibility and variance and is not automatically powered for that threshold.

Report paired completion outcomes, per-attempt acceptance-fraction and quality differences, and paired wall/token/channel costs, including all failures. A paired exact sign/McNemar analysis may describe completion differences; report a paired confidence interval with its declared small-sample method, and do not count hidden test cases or repeated executions of the same artifact as independent trials. With five paired attempts on one public project, intervals describe only within-project development variability when independence is justified; they do not estimate variation across tasks. Plan any larger task-generalization study separately, using development-only variance assumptions before generating its held-out tasks.

A timeout, crash, degeneration abort, cancellation failure or exhausted budget is an unsuccessful development attempt and remains in the completion denominator; retain its partial artifact and costs. Forbidden access, leaked hidden evidence, scorer tampering or undisclosed rescue makes evidence INVALID, with the cause and all attempted runs reported rather than quietly replaced. If an external infrastructure incident invalidates a pair, apply a predeclared symmetric disposition and show the original costs; no discretionary rerun follows an unfavorable score. A cancellation failure also blocks subsequent host use until termination is established.

Engineering readiness remains PASSED/FAILED/BLOCKED/INVALID and is separate from the series' final scientific verdict: **Demonstrated improvement**, **Failed challenge**, **Inconclusive**, or **Invalid experiment**. A precise tie can be a failed challenge under the must-improve contract; inadequate precision is inconclusive, and a pilot advantage alone does not establish broad superiority. Preserve a contrary result if normal Duo wins.

No Test A question, answer, activation, per-item score or held-out failure is available for task construction, prompt tuning, translator selection or budget calibration. Test A's aggregate lookup result motivates a hypothesis only; these project outcomes require fresh evidence. Before launch, the remaining launch record must resolve the Drift product treatment, approved cap/host window, independent evaluator and whether the no-link allocation is included; no launch is implied while any prerequisite remains blocked.
