# M0a adapter research — Qwen3.8-Flash-Next × GLM-5.3-Flash

**Date:** 18 September 2026 · **Stage:** M−1 / M0a preparation · **Type:** source research only.
**No model was loaded or executed** (owner instruction). Every statement below comes from
reading configs, model cards and modeling source; nothing here is a parity or transfer
result. Items marked *to verify* need a test once execution is authorized.

Sources read:
- `transformers==5.17.0` `models/qwen4_exp/modeling_qwen4_exp.py` (2,745 lines) and
  `models/glm5_next/modeling_glm5_next.py` (2,426 lines), installed in `.venv-next`
  (separate from the pinned reference `.venv`).
- Checkpoint configs and cards on `spark-a.invalid` (read-only): `RadixArk/Qwen3.8-Flash-Next-NVFP4`
  @ 7b71922, `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ 25a44fd.
- oMLX (github.com/jundot/omlx, local @ 775a380a; app 0.7.0.dev1) vendored MLX code:
  `omlx/patches/mlx_vlm_{qwen4_exp,glm5_next}_compat/`.
- Hugging Face model metadata (sizes, revisions, gating) and mlx-lm 0.31.3 model list.

## 1. Architecture facts that decide the adapter

| | Qwen3.8-Flash-Next (model A) | GLM-5.3-Flash (model B) |
|---|---|---|
| `model_type` | `qwen4_exp` (VLM wrapper; text `qwen4_exp_text`) | `glm5_next` (VLM wrapper; text `glm5_next_text`) |
| Layers | 48 | 45 |
| KV-bearing layers | 12 full attention: indices 3, 7, …, 47 | 11 DSA/MLA: indices 3, 7, …, 43 |
| Other layers | 36 Gated DeltaNet (linear, recurrent) | 34 Kimi Delta Attention (linear, recurrent) |
| Attention heads | 24 query, 2 KV (GQA 12:1), head_dim 256 | 64 heads, MLA latent 512, qk_nope 256, v 256 |
| Positional encoding | M-RoPE, partial: 64 of 256 dims rotated | **None (NoPE)** in main attention and indexer |
| Sparse selector | QSA indexer: 4-token blocks, budget 2,048 tokens | DSA indexer: 4-token k-pools, top-k 2,048 |
| Residual | 4 hyper-connection streams (gated residual) | 4 mHC streams (Sinkhorn-projected mixing) |
| MLP | MoE 512 experts, 10 active | MoE 288 experts, 8 active, 1 shared; first 3 dense |
| Extra | PLE: hashed n-gram embedding (~95 GiB BF16) into every stream | — |
| Total size | ~180B params (360 GB BF16 source) | larger (FP8 official 306 GiB; BF16 599 GiB) |

## 2. Canonical KV boundary (spec §4.1, D4)

**Qwen3.8 (per full-attention layer):**
`K = k_norm(k_proj(x))` reshaped `[T, 2, 256]`, before RoPE; `V = v_proj(x)` `[T, 2, 256]`,
never rotated. `k_norm` is Qwen's `(1 + w)` RMSNorm. Rotary applies to the first 64 dims
only (`partial_rotary_factor 0.25`), via M-RoPE sections `[11, 11, 10]`. For text-only
input all three position streams are equal, so it should reduce to ordinary 1-D RoPE on
64 dims (*to verify*). Scale `256^-0.5`. The attention output is multiplied by
`sigmoid(gate)` (second half of `q_proj`) before `o_proj`.
FP16 payload: 12 layers × 2 (K,V) × 2 heads × 256 × 2 B = **24 KiB/token**.

**GLM-5.3 (per DSA layer):**
Canonical entry is the normalized MLA latent `c = kv_a_layernorm(kv_a_proj_with_mqa(x))`,
`[T, 512]`. There is no rope component (`qk_rope_head_dim = 0`). Per-head K and V are
*derived* by the receiver's own `kv_b_proj` (K_nope 256 and V 256 for each of 64 heads).
The stock transformers cache stores the expanded K/V (64 × 256 × 2 per token). oMLX caches
only the latent, which matches this boundary. Scale `256^-0.5`.
FP16 payload: 11 layers × 512 × 2 B = **11 KiB/token**.

**Consequence:** the kit's `KV(k, v)` with `[T, H, D]` K and V of equal shape fits Qwen but
not GLM. The layout must become a per-model declared descriptor (`kv_split` vs
`mla_latent`), carried in the pinned session contract. Wire v1 (K and V blocks per layer)
needs a v2 that encodes a latent-only layer. This is consistent with HIVE_MIND H1.

## 3. Private native state beyond KV (needed for identity import and checkpoints)

| Qwen3.8 | GLM-5.3 |
|---|---|
| 36 × DeltaNet conv state (kernel 4) + recurrent state `[48, 128, 128]` fp32 | 34 × KDA conv state + recurrent state fp32 |
| 12 × rotated K/V cache + QSA indexer raw keys | 11 × latent cache + DSA indexer packed keys/gates |
| PLE layers: short-conv state + last `ngram_size − 1` token IDs | — |
| Full 3-stream position IDs bound to the cache (indexer needs them) | — |

M0 identity replacement (§4.2) therefore means: canonical KV for every KV-bearing layer
**plus** an exact copy of every recurrent, conv, indexer and PLE state. Recurrent states are
not per-token and cannot be streamed as deltas. They stay private and are never bridged.
Foreign-memory append touches the KV-bearing layers only.

## 4. Sparse indexers and foreign entries

Both indexers use their own separately projected keys (Qwen `index_qk_proj`, GLM `wk`),
which a foreign model does not produce. Options:

1. **Foreign bank bypasses the indexer (proposed).** Native tokens go through the stock
   indexer unchanged. Foreign entries form a separate, always-visible, gated block inside
   the same softmax (§4.5). The bank is bounded (4 sinks + 128 recent, D18), so the added
   cost is bounded. This keeps the hard-off gate an exact native-only path.
2. Let the indexer score foreign entries. This needs foreign *indexer* keys, which would be
   another learned projection. Deferred; it would be an ablation, not the baseline.

Below about 2,048 visible native tokens, both indexers select every token (Qwen: at most
512 blocks of 4 plus the tail; GLM: 2,048 / 4 pools plus the tail). Short M0 tests
therefore exercise dense attention (*to verify*); long-context qualification must cover
the selecting regime.

## 5. Positions (§4.4, D16)

- **GLM as receiver:** NoPE. Foreign entries get no rotation, and D16 recency positions
  have nothing to act on. Order and recency information is not expressible through phase.
  If it matters it must come through the projection or an explicit learned recency
  feature. That feature would be a new, declared, ablated sidecar (proposed decision).
- **Qwen as receiver:** apply D16 virtual positions through Qwen's own rotary to the first
  64 dims of projected K only. V is never rotated.
- **Qwen → GLM direction:** canonical Qwen K is pre-RoPE, so there is nothing to de-rotate.

## 6. Attention with foreign entries

**Qwen receiver:**
`out = sigmoid(gate_q) · (P_nat V_nat + P_for V_for)`, with
`P = softmax([S_nat + mask_nat + indexer_mask, S_for + mask_for + log g])`, GQA 12:1 for
both native and foreign KV. The output gate is the model's own and applies to both parts.

**GLM receiver:** foreign latent `c_f → K_f, V_f` through the receiver's frozen `kv_b_proj`
(or the absorbed `embed_q` / `unembed_out` form oMLX uses). The same shared-softmax equation
applies per head. The expansion reuses frozen receiver weights and adds no trainable
parameters.

## 7. BridgeProjector shape (§4.3)

- **Qwen → GLM:** `(K [2×256], V [2×256]) → c [512]`. MLA packs K and V information into
  one latent, so "separate K and V maps" cannot hold for this direction. Proposed: a joint
  map into the latent, reported as a declared deviation from §4.3.
- **GLM → Qwen:** `c [512] → K [2×256]` and `c [512] → V [2×256]` as separate maps. This
  keeps the §4.3 rule.
- Ridge remains closed-form in both directions. Layer pairing is 12 ↔ 11 and is chosen from
  training/development data only (M−1.3).

## 8. Runtimes, loaders and quantization

| Checkpoint | Size | Loads in stock transformers 5.17? | Known runtime |
|---|---|---|---|
| Qwen/Qwen3.8-Flash-Next (BF16) | 335.3 GiB | yes (architecture present) | — |
| Qwen/Qwen3.8-Flash-Next-FP8 | 172.8 GiB | likely via fine-grained FP8 (*to verify*) | — |
| RadixArk/…-NVFP4 (on spark-a.invalid/spark-b.invalid) | 125.9 GiB | **no**: ModelOpt format; transformers NVFP4 is on-the-fly only | SGLang (card's eval used sgl-eval) |
| Vontra/…-MLX-4bit-MTP | 105.5 GiB | n/a (MLX) | oMLX `qwen4_exp` patch |
| zai-org/GLM-5.3-Flash (official) | 305.8 GiB | yes | — |
| Mia-AiLab/…-EXL3-4bpw (on spark-a.invalid/spark-b.invalid) | 164 GiB | **no**: EXL3 trellis codebooks | custom vLLM, TP2, FP8 MLA cache (SM120 image) |
| RadixArk / RedHatAI …-NVFP4 | 184–189 GiB | no / compressed-tensors (*to verify*) | vLLM / SGLang |
| orcarouter/GLM-5.3-Flash-MLX | 2-bit 135, 3-bit 171.6, 4-bit 190 GiB | n/a (MLX) | oMLX `glm5_next` patch |

Upstream mlx-lm 0.31.3 supports neither architecture. oMLX vendors both. The spec (P1 §7)
says not to do the science on paged serving caches (vLLM/SGLang). The dedicated adapter
owns its state instead.

## 9. Qualification is not possible in float32

Neither model fits any host at FP32 or BF16 (Studio 256 GB; Spark 128 GB unified;
MacBook 128 GB). D13's "float32 PyTorch reference first" cannot be followed for this pair.
Proposed three-level M0a qualification:

1. **Architecture parity, FP32, tiny random configs:** dedicated adapter vs stock
   transformers 5.17 forward, with D19 tolerances (2e-5). This checks operator boundaries,
   hybrid state, indexer, hyper-connections and the exact hard-off path. It runs on this
   MacBook without real weights. *Needs execution authorization.*
2. **Real-weight parity in deployed precision:** the adapter on the MLX (and/or CUDA)
   checkpoint vs that runtime's stock forward on the same weights, with a
   **preregistered** lower-precision tolerance plus logit-distribution analysis (D19
   allows this; the tolerance must be set before looking).
3. **Cross-runtime fixtures** (§10.1) once both sides exist.

## 10. Placement options (memory only; nothing tested)

| Option | Model A (Qwen) | Model B (GLM) | Fits? | Issues |
|---|---|---|---|---|
| 1 (spec-shaped) | Studio, MLX 4-bit 105.5 GiB (oMLX) | spark-a.invalid + spark-b.invalid, EXL3 164 GiB split across hosts | plausible | Studio disk has 85 GiB free; needs an EXL3 CUDA adapter on GB10 (SM121, unverified); two-host pipeline for one model; both Sparks currently run `v41_stage_server` |
| 2 | one Spark, NVFP4 126 GiB with PLE tables off-GPU (*unverified*) | Studio, MLX 3-bit 171.6 or 4-bit 190 GiB | tight | Qwen NVFP4 needs a ModelOpt-format CUDA loader; PLE offload unverified |
| 3 | Studio MLX 105.5 | Studio MLX 2-bit 135 | no | ~240 GiB of weights on 256 GB before caches |

## 11. Deviations from the spec that need owner sign-off

1. D14 replaced by the owner's pair (recorded as OD1); M0b waived (OD2).
2. D13: no FP32 real-weight reference is possible; adopt the three-level qualification (§9).
3. Reference adapter admission rules (no quantization / hybrid / MLA): replaced by two new
   dedicated adapters with their own qualification.
4. §4.3 separate K/V maps: not possible into an MLA latent (Qwen → GLM direction).
5. D16 recency positions: not applicable to a NoPE receiver (GLM).
6. Wire v1 K/V layout: needs a latent-capable v2.
7. Foreign entries bypass both sparse indexers (§4, proposed baseline).

## 12. Open questions for the owner

1. Which placement option (§10), and when can the Studio disk and the Sparks be used?
2. Approval to *execute* level-1 parity tests on tiny random configs (no real weights, no
   downloads, CPU only, minutes).
3. Preregistered lower-precision tolerance policy for level-2 parity.
4. Sign-off on the deviations in §11.
5. Still open from M−1: budgets, `SERIES.md`, sandbox identities.
