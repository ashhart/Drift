# Model adapters and the model registry

**Status:** adapter contract, registry and plugin loader implemented; each real checkpoint still requires its own qualification evidence.
**Goal:** Drift is model-agnostic. A model joins by bringing its adapter; nothing
else in Drift changes.

## Principle

Drift's core never mentions a model family. The bank, gate, scheduler, transport,
replay, audit, experiments and the hive pool work only with what an adapter declares.
Everything model-specific lives in two places:

1. **Adapter code**, one per *architecture and runtime*, for example `qwen4_exp/mlx`,
   `qwen4_exp/torch`, `glm5_next/mlx`. It knows where the cache is, how attention works,
   and how to add foreign entries to it.
2. **A registry entry**, one per *model checkpoint*. It names its adapter, pins the exact
   checkpoint and holds that model's qualification evidence and fitted translators.

A second checkpoint of a supported architecture needs a registry entry and qualification,
but no new code.

Third-party Python packages can register loaders through the `drift.adapters` entry-point group.
`drift adapter scaffold` creates the package boundary but leaves the architecture code explicitly blocked.

## Adapter contract

Every adapter implements the same interface. The toy `FrozenDecoder` becomes the first
implementation.

| Member | Purpose |
|---|---|
| `describe()` | Layout descriptor: model type, cache-bearing layer indices, per-layer canonical layout (`kv_split [T,H,D]` or `mla_latent [T,C]`), positional scheme (`rope`, `partial_rope`, `none`), attention scale, vocab, tokenizer hash |
| `forward(ids, state, foreign, gates, override)` | One step on the frozen model. Returns logits, new private state, canonical capture for new tokens, foreign attention mass |
| `capture` | Canonical entries at the declared boundary (after native key norm, before rotary; values unrotated; MLA latent after its norm) |
| `attend_foreign` | Adds a separate, gated foreign block inside the model's own softmax. `override=0` must be the exact native path |
| `rephase(entries, positions)` | Applies the model's own positional scheme to foreign keys. Identity for NoPE models |
| `snapshot_state / restore_state` | Private state beyond the cache: recurrent, conv, indexer and n-gram state. Needed for identity import, checkpoints and replay |
| `import_self_prefix` | M0 identity replacement only |
| `frozen_digest()` | Proves backbone weights are unchanged |

Rules:
- This adapter contract is separate from the native serving connectors, which have their own qualified cache-write boundaries.
- Sparse selectors (indexers) stay native-only. Foreign entries bypass them as a bounded,
  gated block (research note §4).
- An adapter may refuse inputs it has not qualified, and must refuse explicitly.

## Qualification suite (the same for every adapter)

An adapter is usable only after it passes, with recorded evidence:

1. Stock parity: adapter vs the architecture's stock forward, full prefill and
   incremental decode, lengths 1, 2, 7, 16, 127, 256 and the deployment length.
2. Exact hard-off: `override=0` and an all-masked foreign bank reproduce native output
   and native state bit-for-bit.
3. Capture boundary: canonical keys are pre-rotary and post-norm; values unrotated.
4. Identity import: self-prefix plus restored private state matches ordinary decoding.
5. Frozen weights: digest before equals digest after.
6. Precision: FP32 tolerance 2e-5 (D19) on tiny random configs. Lower-precision real
   weights need a tolerance preregistered before the run.

## Registry entry (one per model)

```text
registry/<model-id>/
  model.json          checkpoint repo + revision, weight/tokenizer/config hashes,
                      adapter id + adapter commit, runtime lock, quantization, host
  qualification/      suite results and hashes for this checkpoint
  translators/        fitted maps: pairwise (Drift) and/or pool.v1 writer/reader (Hive)
```

## Joining

- **Known architecture:** add a registry entry, run qualification on the real checkpoint,
  fit its translator. No code change.
- **New architecture:** write one adapter, pass the suite on tiny random configs, then
  on real weights. Then join as above.
- **Hive mode:** the new model fits only its own writer and reader against the frozen
  `pool.v1` (HIVE_MIND H9). Existing members are untouched.

## First adapters

| Adapter | Status |
|---|---|
| `drift_toy/torch` | exists (kit reference) |
| `qwen3` and `llama/torch` | exists (kit HF adapter; parity tests pass on tiny configs) |
| `qwen4_exp/torch`, `glm5_next/torch` | implemented; native checkpoint/runtime qualification is still required |
| `qwen4_exp/mlx`, `glm5_next/mlx` | implemented; optional MLX environment and separate native qualification required |
