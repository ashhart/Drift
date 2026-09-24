# Current status

As of 23 September 2026. This is the canonical current-status summary; dated
research and engineering records describe the particular runs they document.
Documentation cleanup does not rerun an experiment or change its verdict.

## Headline results

Two different models, Qwen3.8-Flash-Next on a Mac Studio and GLM-5.3-Flash on two
DGX Sparks, exchange memory as translated cache rows over MCDMA. In the tests below,
the fact reaches the other model only that way; prompts, controls and the task text
itself still travel as ordinary text.

| Result | Verdict | Record |
| --- | --- | --- |
| GLM recalls a fact only Qwen was told | SUPPORTED: 24 of 24 from memory, 24 of 24 from text, 6 of 24 guessing | [causal recall confirmation](evaluation/CAUSAL_RECALL_CONFIRMATION.md) |
| GLM writes working code needing a detail only Qwen was told | SUPPORTED: 15 of 16 from memory, 16 of 16 from text, 0 of 16 guessing | [joint code causal confirmation](evaluation/JOINT_CODE_CAUSAL_CONFIRMATION.md) |
| A receiver skips re-reading shared context | SUPPORTED: 4.59 times less prefill for Qwen, accuracy held | [prefill cost](evaluation/PREFILL_COST_RESULT.md) |

The fix that lifted the first two from 16 of 24 and 0 of 16: GLM now computes its
question after the memory lands. Before, it read its whole prompt first and only
its generated words could see the memory.

## Engineering gates passed

| Gate | Result | Record |
| --- | --- | --- |
| GLM given its own attention rows and recurrent state | 24 of 24 answers, 13 identical to text | [GLM own-cache gate](evaluation/GLM_OWN_STATE_GATE.md) |
| Qwen given its own rows and recurrent state | 24 of 24 answers, 21 identical to text | [Qwen own-cache gate](evaluation/QWEN_OWN_STATE_GATE.md) |
| DeepSeek V4 given its own compressed rows | 24 of 24 against native text's 23; exact writes on both ranks | [DeepSeek V4 gate](evaluation/DSV4_DRIFT_GATE.md) |

GLM and DeepSeek V4 on the Sparks start with their Drift connectors through systemd
drop-ins. Qwen3.8 on the Sparks has a connector that has not yet passed its live gate;
on the Studio, Qwen runs the loop's own MLX path.

## Exploratory, not yet confirmed

- Drop-in on a real project: a fresh Qwen attaches to GLM's cache of this repository's
  own modules, pulled over MCDMA at about 49 Gbit/s, and answers without reading the
  project. It prefills about 96 tokens instead of the whole context. With its own cache in
  place of GLM's it matches text on every set. With GLM's cache read by a contextual reader
  fine-tuned on answers, it matched text on this repository's 32 short-context questions,
  answered 69 of 70 long-context questions over MCDMA against text's 70 with its first token
  4.7 times sooner, and matched text on 40 questions about code from other packages (35
  each). It trails text on a held-out project, by 2 to 4 of 40 questions, and on Stockledger,
  where its modules average 29.0 of 33 checks against text's 30.5:
  [drop-in results](evaluation/DROPIN_REAL_PROJECT.md), [held-out project](evaluation/HELD_OUT_PROJECT.md).
- Payoff at long contexts: five joiners over 32k, 64k and 128k tokens of code. A joiner
  attaching GLM's translated cache had its first token 66 to 131 times sooner than one
  reading the text (2.9 s against 381 s at 128k), prefilling 92 tokens instead of the whole
  context; five of them were all answering in 119 s at 128k, counting GLM's read, the
  pull and the translation, where five reading the text took 1,904 s. Every way found its
  needles: [payoff](evaluation/PAYOFF_LONG_CONTEXT.md).
- Shared space: one command adds a model to a GLM-anchored hub, checks both maps and runs
  the drop-in test; GLM to Qwen through it answered 29 of 32 against 26 for the direct
  linear translator it replaces: [shared space](evaluation/SHARED_SPACE_MEMBER.md),
  [guide](guides/SHARED_SPACE.md).
- MCDMA recovers from an orphaned or stalled pull on the next pull, byte for byte, with no
  daemon restarted; the Studio daemon must run inside a live session, since macOS blocks a
  process's new LAN connections once its session ends: [recovery](evaluation/MCDMA_RECOVERY.md).

- Qwen to GLM on code: a reverse contextual reader over Qwen's rows let GLM answer all 32
  short-context code questions from translated rows alone, as text did. On unseen code it
  mostly refused (1 of 5 against 5), and it cannot yet be fine-tuned on answers, since GLM
  runs only inside vLLM. GLM's live receiver takes at most 4,096 rows and 128 MiB per
  publication, and past either limit it stops its engine rather than the request:
  [six directions](evaluation/SIX_DIRECTIONS.md).
- A recurrent state translated from Qwen lifted GLM from 20 to 24 of 24 on gate
  questions, where text scored 23: [development results](evaluation/TRANSLATED_STATE_DEVELOPMENT.md).
- GLM to Qwen: from latents of GLM reading, translated rows plus a translated state
  named GLM's recommendation in 22 of 22 cases against 18 for text. From latents GLM
  writes in the live loop, which an offline replica now reproduces exactly, Qwen names
  GLM's pick in about 60% of 48 cases against text's 98%. Replaying a live run's saved
  taps gives the live answers, so the loop is faithful; translation fidelity limits it.
- DeepSeek V4's cache pages decode, and first linear translators from its grouped
  entries let Qwen and GLM answer 2 and 3 of 20 gate questions:
  [development results](evaluation/DSV4_TRANSLATOR_DEVELOPMENT.md). Into DeepSeek, through
  the shared space, GLM's and Qwen's caches answered 1 and 0 of 20 where DeepSeek's own rows
  answered 20: [six directions](evaluation/SIX_DIRECTIONS.md).

## Earlier engineering results

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

- Native qualification of `/drift subagent enable`, stock Duo room integration and arbitrary nested subagent workflows; the new command passes installed-OMP synthetic QA but requires owner-provisioned services and profiles.
- GLM to Qwen at text level in the live loop, and any translated-state claim on passages the translators never saw.
- Usable translators between DeepSeek V4 and the other two models, either way; linear maps between per-token caches and its grouped entries fall short.
- A held-out run on names and seeds the owner writes, per the [held-out protocol](evaluation/HELD_OUT_PROTOCOL.md).
- General two-way recall reliability, source attribution and long-context quality.
- Complete final-tail coverage and repeated-run reliability; an earlier control closure remains unattributed.
- Portable two-host tap manifests and other model families without their adapters, trained translators and qualification evidence.
- The formal M-1 through M6 research gates merely on the strength of these engineering runs.

The shell CLI creates tap projects using supplied translators; it does not train
new translator weights. `/drift` defaults to the reference service; the
[restricted KV-only counterpart](guides/DRIFT_SUBAGENTS.md) does not convert an
existing Duo room into a native KV-linked pair or provision its servers.

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
