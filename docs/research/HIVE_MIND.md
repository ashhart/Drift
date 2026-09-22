# Hive Mind — a planned stage on top of Drift

**Status:** PROPOSED / NOT STARTED. Owner-approved direction, 18 September 2026.
**Depends on:** Drift M1 cross-family evidence (see "Entry gate").
**Relationship to the spec:** additive. Nothing here amends M−1 through M1 of
`TELEPATHY_AGENT_ENGINEERING.md` except the three interface decisions H1–H3, which
change code shape, not stage order, evidence, or claims.

## 1. Idea

Drift gives each receiver a private bank of *projected foreign* KV from one
sender. The Hive Mind replaces the per-pair bank with **one logical, append-only KV
pool** that any enrolled model can write to and read from. Each model owns a
**writer** (native canonical KV → pool format) and a **reader** (pool format →
its native attention format). N models need 2N converters, not N(N−1) bridges.

The payoff is **enrollment and swap-in**: once a model's writer/reader are fitted
to the frozen pool format, it can join a running hive and immediately attend to
everything already written, without rereading the source text. Entries outlive
their writer: a model swapped out leaves its contributions readable to others.

The accurate claim is "reads what the hive has written, through two lossy
conversions", measured against rereading the text. It is not "knows everything":
the pool is attention-accessible memory, not weight knowledge, and it is bounded.

## 2. Drift is the two-member special case

| Mode | Pool format | Writer | Reader |
|---|---|---|---|
| Drift (M0–M1) | sender's native canonical KV | identity | direct sender → receiver map |
| Hive | shared format `pool.v1` | model → `pool.v1` | `pool.v1` → model |

Everything else is shared: adapters, gated foreign attention, PositionAligner,
atomic publication and pinned views, one-epoch-lag scheduling, transport, replay,
audit isolation, E1–E4.

## 3. Interface decisions made now (apply from M0)

**H1 — Pool format is a declared parameter.** Every bank/publication carries a
`pool_format` from the pinned session contract: `native:<model-manifest-hash>` for
Drift, `shared:<format-version>` for the hive. Code paths do not branch on it
beyond selecting converters.

**H2 — Multi-writer, append-only bank.** Entries are keyed by controller-assigned
`(writer_id, sequence)`; writer IDs are fixed enums from the run manifest, never
model text. Writers only append complete all-level publications; no writer can
mutate or delete another's entries. Expiry/eviction is a controller policy. The
reference `Delta` stays free of prompt strings, token IDs and answer labels.

**H3 — Projector split into writer and reader halves.** `BridgeProjector` becomes
the composition `reader ∘ writer`. In Drift mode the writer is the identity and
all learned capacity sits in the reader, which is exactly the existing pairwise map.

## 4. Hive-stage rules

**H4 — Private native caches remain.** Each model keeps its full private native
cache. The pool is a separate, gated set of attended entries (the spec's
foreign-memory append path, D2), never a replacement for native state.

**H5 — No self-reads through the pool.** A reader excludes entries with its own
`writer_id`; it already has them natively. This avoids lossy self-echo and the most
direct feedback loop.

**H6 — Depth levels replace layer pairs.** The pool has L levels (initially 12).
Each model declares a level map from its KV-bearing layers, chosen from training
and development data only. For the founding pair, Qwen3.8-Flash-Next has 12
full-attention layers; GLM-5.3-Flash has 11 compressed-latent (MLA) layers.

**H7 — Pool positions are pool age.** Entries are stored unrotated. Each reader
applies its own rotary from pool order using the D16 recency policy, generalised
from "source position" to "pool sequence". Readers with no rotary on the cached
component apply none.

**H8 — Causality is unchanged.** Epoch k reads the pool as committed at k−1 (D10).
A step pins one complete pool view for every bridged level.

**H9 — Frozen format and enrollment.** The founding pair trains `pool.v1` jointly.
After the entry gate the format is frozen and versioned. Later models are enrolled
by fitting only their own writer/reader against the frozen format, using a
declared calibration corpus with no evaluation answers. Enrollment never changes
another member's converters. A new format version requires an explicit pool
migration and requalification of every member.

**H10 — Physical layout.** One logical pool; each reader keeps a local, already
projected replica and fetches only new entries (D1, §13.1). No reader re-reads the
whole remote pool per step.

## 5. Training

Writers and readers are small sidecars; backbones stay frozen (D7). Objectives:

```text
Round trip:  reader_A(writer_A(kv_A))  ~ kv_A              (self-reconstruction)
Cross:       reader_B(writer_A(kv_A))  ~ kv_B on aligned text (causal endpoints, D8)
Behavior:    KL(receiver with own context || receiver with pool entries), receiver vocab
Enrollment:  new member's writer/reader only; pool.v1 and other members frozen
```

Every term, budget and stopping rule is preregistered and ledgered separately.

## 6. Entry gate

Start the Hive stage only when Drift M1 shows a preregistered cross-family
effect over the no-context floor with the direct map. If the one-hop direct map
cannot beat the floor, a two-hop shared format is not expected to, and the result
is recorded as the reason not to proceed.

## 7. Experiments

- **H-E1 Pool handoff.** A writes a document into the pool; B answers from the
  pool only. Floor: no context. Ceiling: B rereads the text. **Baseline: the
  Drift direct map on the same units.** A hive result below the direct map is
  a valid finding.
- **H-E2 Swap-in.** Build a pool with the founding pair, enroll an unseen third
  model, and have it answer held-out questions about pool contents without the
  text. Report against its reread ceiling and no-context floor.
- **H-E3 Writer departure.** Remove a writer; measure how well the remaining
  members still use its entries.
- **Converter quality table.** Round-trip and cross reconstruction plus behavior
  KL for every writer/reader pair.

## 8. Risks

- **Bottleneck loss.** A shared format may discard pair-specific information.
- **Feedback amplification.** Multi-writer read/write loops raise echo, drift,
  gate-saturation and storm risk; the M2 degeneration detectors are mandatory.
- **Joint training difficulty.** More moving parts than a closed-form ridge map.
- **Attention cost.** Pool reads cost compute each step; both founding models have
  top-2048 sparse indexers that must be taught to select pool entries or the
  entries are never attended.

## 9. Build order

1. M−1 → M1 as specified, with H1–H3 in the code shape.
2. Entry gate (§6).
3. Train `pool.v1` with the founding pair; H-E1 against the direct map.
4. Freeze `pool.v1`; enroll a third model; H-E2, H-E3.
5. M2+ use whichever mode the evidence supports.
