# Drift: engineering plan

Two models, one project, one shared mind. Model A runs on the Studio, model B runs on a Spark, both work the same repo the way the OMP Duo room already does, except they never speak in tokens. Each reads the other's KV cache directly across the MCDMA link, each can leave a question inside its own cache for the other to find. Pure drift, on hardware you already built the pipe for.

The bridge layer inside keeps the anatomical name, the callosum, after the nerve that links the brain's hemispheres. The project is called what it is.

Related work that must be cited and beaten, not rediscovered: Cross-Model KV Cache Transfer in LLM Families (arXiv 2608.03893, Aug 2026, same-family prefill reuse, closed-form linear maps) and the Bicameral Model (Flamant, May 2026, two frozen same-model 0.5B streams coupled by a 6.2M-parameter interface). Neither is cross-family. Neither is at real scale. Neither has a public demo, a mailbox, or a head-to-head. That is the delta.

## 1. What exists and what is new

| Piece | Status | Role here |
|---|---|---|
| MCDMA | Built | Zero-copy memory windows between Studio (Metal) and Sparks (CUDA). The data plane. |
| OMP Duo plugin | Built | Two models, one repo, negotiated split, ASK/ANSWER over hub messages. The control plane and the text-only baseline arm. |
| drift-core | New | Activation taps, foreign-KV injection, bridge projectors, gates. The data plane's brain. |
| drift-mailbox | New | Questions and answers as pure thought, parked in KV. The novel protocol. |
| drift-eval | New | Drift, organism, mailbox-duel experiments plus degeneration detectors. The scorer. |
| omp-drift | New plugin | The room UI: audit view, mailbox latency board, run control. |

Sharp boundary to keep the claim honest: Duo cannot touch activations. It coordinates models through OMP's provider APIs. Everything below the coordination layer is new code, and the text-channel Duo room remains the baseline the drift arm must beat or match.

## 2. Architecture

```
[ Studio, Metal/MLX ]                      [ Spark, CUDA ]
  model A (frozen)                           model B (frozen)
   attention hooks ──┐                        ┌── attention hooks
                     │                        │
   ForeignKVInjector │   MCDMA zero-copy      │ ForeignKVInjector
   BridgeProjector <─┼──── memory windows ────┼→ BridgeProjector
   ThoughtWriter     │   (fallback: TCP)      │ ThoughtWriter
   Gate/Sync         ┘                        ┘ Gate/Sync

        control plane: OMP Duo room (hub messages, todo board, audit log)
        data plane:   KV tensors over MCDMA, no tokens, ever
```

Two planes, cleanly separated. The control plane stays OMP: room lifecycle, task split negotiation at session start (that can be text, it is setup), the audit log, the scoreboard. The data plane is every thought after the handshake: KV entries flow across the wire, nothing else does.

Hardware mapping. Model A and its projector kernels on the Studio GPU reading Spark-resident KV through the MCDMA window; model B mirrored on the Spark reading Studio KV the same way. If mutual reads through the window measure slower than expected, the transport layer swaps to async copy without changing any science code. The in-process backend runs both models on the Studio alone for development.

Model choices, cross-family by design: Qwen3-8B and GLM-4-9B or Llama-3.1-8B, whichever pair you stock locally, plus one same-family pair (Qwen3-4B to Qwen3-8B) reserved purely for the M0 replication. Different tokenizers are mandatory. Identical tokenizers invalidate the cross-family claim.

## 3. Component specs

### drift-core

`AttentionTap`: forward hooks on every attention block's KV path. Captures per-layer K and V after RoPE is applied (and again before, see PositionAligner). No model weights change, ever.

`ForeignKVInjector`: extends the attention context with projected foreign entries before the softmax. Three knobs, all logged: which layers are bridged (start with 8 of 32), the foreign window (sliding recent window plus sink entries, StreamingLLM-style), and whether injection is read-only (A observes B) or bidirectional.

`BridgeProjector`: per-layer learned map from A's (n_kv_heads_A, head_dim_A) to B's (n_kv_heads_B, head_dim_B). MLP with layer-norm, a few million parameters total, same spirit as the Bicameral interface but cross-family. GQA head-count mismatch handled by a learned head-mixing matrix, not naive repeat-interleave, and the naive version is kept as a reported baseline because the same-family papers claim closed-form linear maps suffice. If a ridge-regression map gets 90 percent of the learned projector's accuracy, that is a finding, publish it as one.

`PositionAligner`: the nastiest correctness detail. Foreign entries carry A's rotary phase. Two strategies, both implemented: re-encode foreign entries at B's RoPE base with A's relative distances preserved, or learn a phase remap. Attention sinks (the first few tokens models lean on) are never dropped and never reordered.

`Gate`: per-layer learned scalar gate on foreign attention weight, initialized near zero and trained open. This is the firewall against thought contagion. The ARC trials showed naked models imitate whatever frame pattern dominates their input; the gate is the mechanical answer, and its per-layer trajectory during training is itself a chart worth posting.

`SyncController`: lockstep, chunked, or sliding modes. Bandwidth budget below. Full-cache sync only at handoff events; steady-state sync is per-step deltas.

### drift-mailbox (the novel part)

A question is a thought aimed at someone. Mechanics:

1. `ThoughtWriter` forces a short token sequence ("Q: does the parser handle nested unions?") through model A's input embedding path while suppressing emission to any user-facing stream. Those tokens now have live KV entries in A's cache. A has not spoken. A has thought at B.
2. A learned marker entry (one soft token, trained for salience) prefixes the thought so B's attention treats it as addressed mail rather than background reasoning.
3. The bridge projects the marker plus the thought into B's cache. B attends, reads, and answers the same way: B thinks the answer at A. The conversation never becomes text.
4. There is no delivery step, because nothing is ever sent. Reception is attention: B thinks the answer, the answer lives in B's cache, and A, already attending to that cache, simply knows, the way you notice a friend nod without being mailed a nod. So the measurable event is noticing, and it has a causal test: track A's attention mass on the foreign marker and answer entries, then ablate those entries and verify A's behavior reverts to the pre-answer distribution. If masking the answer makes A act uninformed, the thought demonstrably crossed. Metrics: noticing rate (fraction of answers A demonstrably used), incorporation latency (decode steps between the answer entering B's cache and A's behavior reflecting it), and ablation delta (how much A's output depends on the foreign entries).
5. Starvation watchdog, reframed: if A's attention never lands on a marker, or lands but incorporation never follows, the mailbox logs an unnoticed question. Unnoticed mail is a first-class metric, not an anomaly. Attention without incorporation is its own interesting failure: the model glanced at the letter and did not read it.

Delivery verification stays outside the minds: `ProbeHead`, a small linear probe trained per model, reads each cache on the experimenter's side and reconstructs what thoughts are present, purely for the audit log. The probes see the caches; neither model sees the probes' output. This keeps scoring external per the series rules while the collaboration channel itself stays token-free.

Failure modes to instrument from day one: echo lock (both models re-attending each other's marker in a loop, counted by marker-attention oscillation), question storms (mailbox fill rate exceeding drain rate), and silent starvation.

### drift-train

Alignment corpus: real repo-shaped data, since the deployment scenario is two models building one project. Code, docs, issues, and genuine Q&A pairs over both models' tokenizers in parallel. Losses: projected-KV regression against the receiver's own activations on the same text, plus a behavior-matching term, KL between receiver-with-injected-KV and receiver-with-own-KV on next-token distributions. Backbones frozen throughout. Training budget capped and reported per the series accounting rules. The projector is the only trainable thing, so this fits comfortably on the Sparks in an afternoon per run.

### drift-eval

Four experiments in escalation order, each with a pre-registered pass bar:

E1 Handoff. A reads a document, B inherits only the bridged cache, B answers. Floor: no context. Ceiling: B re-reads the text. Baseline: equal-length English summary. The transplant demo.

E2 Drift. B holds a secret A never saw as text. Coupled live, can A answer questions about it? Pass bar: statistically above the no-context floor over 100 held-out secrets. This is the clip that travels.

E3 Organism. Reasoning suite: A solo, B solo, coupled pair, and the pair talking in plain English with the same compute. If coupled beats max(solo A, solo B) it is a two-brain organism. If English-chat wins, that is the upset result and it still airs.

E4 Mailbox duel. The Duo scenario end to end: same repo, same task backlog, two arms. Text arm is the existing Duo room with ASK/ANSWER. Drift arm is the coupled room with mailbox only. Score: task completion, wall clock, tokens on the collaboration channel (must be zero, verified by wire capture), noticing rate, incorporation latency, and the ablation delta proving the thoughts were used. The headline if it holds: the pair built the feature without exchanging a single word.

Degeneration detectors run through everything: echo index, cache drift, gate saturation, storm counters. The contagion reel is content by design.

## 4. Transport and the bandwidth math

Per-token KV footprint for an 8B-class model, 32 layers, GQA 8 heads, head_dim 128, fp16: 4 KB per layer, 128 KB per token all layers. Steady-state delta sync at 10 tok/s each way is roughly 2.5 MB/s. MCDMA shrugs at that. Full 4k-context sync at handoff is 512 MB per direction, a one-time cost, and bridging only 8 of 32 layers cuts everything by 4. The transport is not the bottleneck; foreign attention compute is. That is why the foreign window slides and why bridge-layer count is a competitor-tunable knob.

Transport backends behind one interface: inproc (dev, both models in one Studio process), tcp (correctness fallback if the window reads misbehave), mcdma (target, zero-copy, instrumented). The MCDMA swap-in milestone exists to produce the pipe numbers: round-trip KV read latency, projected entries per second, joules per synced thought. Those numbers are an episode on their own for your audience.

## 5. OMP integration

New plugin, `omp-drift`, modeled on Duo's host API usage. It does not replace Duo; it runs beside it as the other arm. Room surface: `/drift start <pair>`, the audit view rendering ProbeHead reconstructions of what each mind currently holds, the mailbox board (delivered, pending, starved, latency histogram), gate trajectories, and `/drift duel <task>` launching the E4 two-arm run with Duo on the text side. Session persistence and restart semantics follow Duo's patterns. The repo, todos, and negotiation-at-start all stay in OMP where they already work.

## 6. Milestones

M0, replication gate (week 1). Same-family pair, in-process, ridge-regression projector, E1 only. Accept: beats the no-context floor, reproducing the published linear-structure claim on your hardware. If this fails, stop and debug the harness before touching anything novel.

M1, cross-family handoff (weeks 2 to 3). Learned projector, mismatched tokenizers and head shapes, E1 plus E2. Accept: E2 above floor. First postable chart either way.

M2, live coupling (weeks 3 to 4). Bidirectional injection, gates, sync controller, E3 with the English-chat control. Accept: gates train stable, echo index bounded, organism comparison measured.

M3, mailbox (week 5). ThoughtWriter, markers, probes, delivery metrics, E4 in-process. Accept: nonzero delivery rate with latency and starvation honestly reported. Even 60 percent delivery is a result; the paper says zero.

M4, MCDMA swap-in (week 6). Transport backend switch, zero-copy windows, pipe instrumentation. Accept: identical E1/E2 scores across backends within noise, plus the latency/joule table.

M5, episode package (week 7). Full duel across the wire, degeneration reel, matrix if a third model joins. Pre-registered bars, receipts post.

M6, competition harness (week 8+). The Invention Trials format: 4 to 5 competitor agents each design projector, gate, sync, and mailbox policies between two frozen models with a fixed training budget. Public development pairs, hidden eval pair. Scoreboard: E1 to E4 plus delivery rate and cost. This becomes the series brief.

## 7. Risks

| Risk | Mitigation |
|---|---|
| RoPE phase mismatch corrupts reads | PositionAligner with both strategies; sink tokens preserved; E1 catches it early |
| GQA head mismatch loses information | Learned head-mixing; report against repeat-interleave baseline |
| Echo/imitation attractors collapse both minds | Gates initialized closed; storm watchdog; the failure itself is logged content, ARC precedent documented |
| Foreign attention compute blows the step budget | Sliding foreign window, bridge-layer subset, chunked sync |
| vLLM-style paged caches make injection miserable | Do not use them. Transformers/MLX with hooks for all science; speed comes from MCDMA reads, not serving-stack tricks |
| Mailbox never delivers | Starvation as a metric; marker salience training; if thought-mail fails, the honest fallback finding is "coupling works, addressing does not", still novel |
| MCDMA window reads add latency | tcp fallback preserves the science; backend equivalence test in M4 |

## 8. Validity and accounting, per SERIES.md

Scorer, probes, and reference answers outside both models' writable reach. Wire capture proves the zero-token claim rather than trusting it. Frozen checkpoints, quantization, seeds, and training budgets declared per run. Repeated trials with variance, pre-registered pass bars, and the four-verdict taxonomy. Development pairs public, evaluation pair hidden and frozen before M5.

## 9. Repo layout

```
drift/
  core/        tap, injector, projector, aligner, gate, sync
  mailbox/     thought writer, markers, probes, watchdog
  train/       alignment data build, projector training, budget ledger
  eval/        E1-E4, detectors, matrix runner, manifests
  transport/   inproc | tcp | mcdma backends, instrumentation
  plugin/      omp-drift
  results/     runs, charts, degeneration reels, episode packages
```

Build order within M1: core with the inproc backend and E1, everything else hangs off those two. The first honest number you want is the ridge-projector same-family E1 score. Everything else is that number, made stranger.
