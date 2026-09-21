# M4.1 MLX / Metal adapters — notes and evidence

**Date:** 18 September 2026 · **Stage:** M4.1 · **Status:** tiny-config qualification done;
no real weights loaded, nothing committed.
**Environment:** `.venv-next` — transformers 5.17.0, torch 2.10.0, mlx 0.32.2, mlx-lm 0.31.3,
mlx-vlm 0.7.1; Apple M5 Max (`applegpu_g17s`), macOS Darwin 25.5.
**Run:** `PYTHONPATH=. .venv-next/bin/python -m pytest -q tests/test_adapters_mlx.py`

New files (nothing existing was modified):

| File | Role |
|---|---|
| `telepathy/adapters/mlx_qwen4_exp.py` | `MlxHybridAdapter` (shared MLX base: forward, state deep-copy, snapshot/restore, identity import, digest) and `MlxQwen4ExpAdapter` |
| `telepathy/adapters/mlx_glm5_next.py` | `MlxGlm5NextAdapter` (imports the base from the Qwen module; hoisting it into `mlx_base.py` is a follow-up because M4.1 could only add these files) |
| `telepathy/adapters/mlx_fixtures.py` | torch → MLX weight transfer, config derivation, `build_pair` / `build_family` / `build_adapters`, `fixture_exchange` / `exchange`, GEMM precision probe, sparse-indexer tie detection |
| `tests/test_adapters_mlx.py` | qualification suite (mirror of `test_adapters_next.py`) + cross-runtime parity |
| this file | decisions and evidence |

## 1. Result summary

| Mode | Result |
|---|---|
| default (`MLX_METAL_GPU_ARCH=applegpu_g16s`, true fp32 Metal GEMM, see §5) | **34 passed, 1 xfailed, 0 failed** (35 tests) |
| Metal default kernels for M5 (`MLX_METAL_GPU_ARCH=applegpu_g17s`, TF32-class GEMM) | **19 failed, 16 passed** |
| torch suite `tests/test_adapters_next.py` (unchanged) | 26 passed |

Observed max abs error, MLX stock forward vs torch stock forward, float32 both sides,
positions before the first indexer tie (§6), lengths 6/7/12/21 (= 1, 2, 7, 16 + 5):

| family | Metal fp32 (default mode) | MLX CPU stream | Metal TF32 (M5 default) | torch fp32 vs torch fp64 (conditioning floor) |
|---|---|---|---|---|
| qwen4_exp | 4.5e-8 (mean 6e-9) | 3.0e-8 | 1.8e-4 (mean 3.8e-5) | 3.8e-8 |
| glm5_next | 7.2e-7 (mean 1.2e-7) | 7.2e-7 | 4.1e-3 (mean 1.0e-3) | 7.2e-7 |

Adapter prefill vs one-token incremental decode (MLX vs MLX, same lengths):

| family | Metal fp32 | CPU | Metal TF32 |
|---|---|---|---|
| qwen4_exp | 4.5e-8 | 6.0e-8 | 1.8e-4 |
| glm5_next | 7.2e-7 (0 for ≤ 8 tokens) | 6.0e-7 | 3.0e-3 (0 for ≤ 8 tokens) |

Tolerances: D19 FP32 tolerance `atol=rtol=2e-5` for every MLX-vs-MLX check and for the
cross-runtime foreign-path parity; cross-runtime stock parity was preregistered at
`atol=rtol=1e-4`, passed, and was **tightened to `2e-6`** (`CROSS_TOL`). Nothing was
widened. In TF32 mode the preregistered 1e-4 fails for both families; the tests are
left failing there (they do not silently adapt), see §7.

## 2. Config mapping (HF → mlx-vlm `TextConfig`)

Both MLX configs are built with `TextConfig.from_dict(hf_config.to_dict())` (fields
filtered by signature), with these decisions:

| Family | Field | Decision |
|---|---|---|
| qwen4_exp | `output_gate_type` | HF stores `None` (= "use `hidden_act`"); the MLX config validates against `{"sigmoid","silu"}`. Resolved as `output_gate_type or hidden_act` (= `silu`), which is exactly how both runtimes' gated norms resolve it. |
| qwen4_exp | `rope_parameters` | HF key `rope_type`; the MLX `__post_init__` renames it to `type` itself. `mrope_section=[1,1,0]` with `partial_rotary_factor=0.25`, `head_dim=8` → rotary dim 2 (one inverse frequency). Text-only positions are 2-D `(B, L)` on the MLX side, for which `compute_mrope_frequencies` collapses the three M-RoPE streams to plain RoPE, matching HF's `recomposition_frequencies` for equal streams. |
| qwen4_exp | `layer_types` | HF normalizes `full_attention` → `qwen_sparse_attention`; MLX does the same. |
| qwen4_exp | `split_ngram_parts`, PLE fields | irrelevant (`ple_layer_ids=[]`); `ple_embed_dim % ngram_heads == 0` validation passes (32 % 16). |
| qwen4_exp | `LanguageModel(config=…)` | `LanguageModel.get_rope_index` (the public text prefill path) reads the *VLM* config for `vision_config.spatial_merge_size` and the image/video/vision-start token ids even for text. A `SimpleNamespace` shim with the upstream `ModelConfig` defaults (248056/248057/248053, merge 2) is passed so the public stock forward runs without instantiating the vision tower. |
| glm5_next | everything | field names coincide; `head_dim` and `qk_head_dim` are recomputed by the MLX `__post_init__`; NoPE is enforced there (`qk_rope_head_dim == 0`). |

## 3. Weight mapping (HF state dict → MLX parameters)

Transfer path: `state_dict()` → temp safetensors (`safetensors.torch.save_file`) →
`mx.load` → key renames below → upstream `sanitize` → strip `language_model.` →
`load_weights(strict=True)` after an explicit missing/extra/shape check.

### qwen4_exp (`Qwen4ExpForCausalLM` → `qwen4_exp.LanguageModel`)

| HF key | MLX key | How |
|---|---|---|
| `model.X` | `language_model.model.X` | renamed to the checkpoint layout `model.language_model.X` first, then the upstream VLM `Model.sanitize` (`sanitize_key`) |
| `lm_head.weight` | `lm_head.weight` | via `sanitize_key` |
| `layers.N.mlp.experts.gate_up_proj` `[E, 2I, H]` | `switch_mlp.gate_proj.weight` / `switch_mlp.up_proj.weight` `[E, I, H]` | upstream sanitize splits at the midpoint of dim −2; verified against HF `Qwen4ExpTextExperts` (`chunk(2)` → gate first) |
| `layers.N.mlp.experts.down_proj` `[E, H, I]` | `switch_mlp.down_proj.weight` | upstream |
| `linear_attn.conv1d.weight` `[C, 1, K]` | `[C, K, 1]` | upstream `moveaxis(2, 1)` |
| everything else (`q/k/v/o_proj`, `q_norm`, `k_norm`, `indexer.*`, `linear_attn.*`, `mlp.gate`, `shared_expert*`, `attn/mlp_hyper_connection.*`, `hyper_connection_mixer.*`, `embed_tokens`) | same names | 1:1 |

The upstream sanitize is called as `Model.sanitize(shim, weights)` with a shim exposing
only `config.text_config`, which is all it reads; the vision tower is never built.

### glm5_next (`Glm5NextTextModel`, headless → `glm5_next.LanguageModel`)

| HF key | MLX key | How |
|---|---|---|
| `embed_tokens.weight`, `norm.weight`, `layers.N.*` | `language_model.model.` prefix | the upstream `LanguageModel.sanitize` keys its fusions on `language_model.model.layers.N` |
| `self_attn.forget_gate.{f_a_proj,f_b_proj,dt_bias,A_log}` | `self_attn.{f_a_proj,f_b_proj,dt_bias,A_log}` | rename; `f_a_proj` then fused by upstream with `b_proj`, `g_a_proj` into `fbg_a_proj` (that order) |
| `self_attn.{q,k,v}_proj` | `self_attn.qkv_proj` | upstream fusion, q,k,v order = HF `torch.cat` order |
| `self_attn.conv1d.weight` `[3·HD, 1, K]` | `self_attn.qkv_conv.conv.weight` `[3·HD, K, 1]` | HF has one depthwise conv over the fused q\|k\|v; upstream only knows separate `q/k/v_conv1d`, so the rename + `moveaxis(2, 1)` is done here |
| `self_attn.q_a_proj` + `kv_a_proj_with_mqa` | `self_attn.qkv_a_proj` | upstream fusion (q part first) |
| `self_attn.kv_b_proj.weight` `[H·(nope+v), C]` | `embed_q.weight` `[H, C, nope]`, `unembed_out.weight` `[H, v, C]` | upstream absorption (§4) |
| `mlp.experts.gate_up_proj` `[E, 2I, H]` | `switch_mlp.gate_proj/up_proj.weight` | split here (gate rows first, verified against `Glm5NextTextExperts._apply_gate`); upstream only knows per-expert keys |
| `mlp.experts.down_proj` | `switch_mlp.down_proj.weight` | rename here |
| `mlp.{gate,up}_proj`, `mlp.shared_experts.{gate,up}_proj` | `gate_up_proj` | upstream fusion |
| `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` | `MoEGate.weight`, `.e_score_correction_bias` | 1:1 (plain arrays) |
| `indexer.{wq_b,wk,k_norm(+bias),weights_proj,index_kpool_compress_ape,index_kpool_compress_gate}` | same | 1:1 |
| `attn_hc/ffn_hc.{fn,base,scale}`, `input_layernorm`, `post_attention_layernorm`, `q_a_layernorm`, `kv_a_layernorm`, `q_b_proj`, `o_proj`, `o_norm`, `g_b_proj` | same | 1:1 |
| — | `lm_head.weight` | the MLX `LanguageModel` always builds a head; loaded with zeros and never used (the adapter returns hidden states like the torch adapter) |

Everything mapped; nothing was dropped. Norm conventions agree on both sides
(Qwen: `(1 + w)` RMSNorm; GLM: `w · x`; gated norms `w · rms(x) · act(gate)`).

## 4. Where the MLX operator boundary differs from HF

### glm5_next
1. **Absorbed MLA.** HF expands the latent per head with `kv_b_proj` (`expand_kv`) and
   caches per-head K/V `[B, H, T, 256]`. mlx-vlm folds `W_k` into the query
   (`embed_q`: `q → q W_kᵀ`, `[B,H,L,C]`), attends against the **latent itself**
   (`[B,1,T,C]` as both K and V) and applies `W_v` afterwards (`unembed_out`). The cache
   stores the latent with a zero-width value block, i.e. the MLX native cache *is* the
   canonical `mla_latent` entry. `native_kv_from_canonical` returns
   `(latent[1,1,T,C], zeros[1,1,T,0])`; the capture test checks `cache.keys == latent`.
2. **Sparse attention by gather, not by mask.** HF builds a boolean mask from the DSA
   top-k over all keys. mlx-vlm gathers the selected latents per query (decode blocks
   ≤ 8 tokens: a per-query loop through `mx.fast.scaled_dot_product_attention`; longer
   prefill: `_sparse_prefill_attention`, whose fused Metal kernel is bf16/fp16-only and
   falls back to a gather + SDPA for fp32). The adapter's foreign path converts the
   top-k indices back into a boolean mask (`topk_to_allowed`, the exact HF
   `build_attention_mask_from_topk`) and runs a dense float32 softmax with the foreign
   columns appended.
3. **Indexer scoring order.** HF: `weights · relu(q·k · scale)`; MLX: `relu(q·k) · (weights · scale)`.
   Identical for `scale > 0`. Pool compression, tail selection and causal candidate
   validity are the same; the top-k tie-break is not (§6).
4. **Projection geometry.** `mlx_vlm.models.linear.linear` runs blocks of 2–8 tokens
   token-wise (GEMV per token, "decode-equivalent") and longer blocks as GEMM; the
   hyper-connection `_mix` does the same. Numerically a reduction-order difference only,
   except on M5 where it decides between fp32 and TF32 (§5).
5. **Hyper-connections.** The fused Metal sinkhorn/collapse kernels are hc_mult = 4 and
   bf16/fp16 only; the tiny config (hc_mult 2, fp32) takes the `_hc_ops` path, which is
   the HF formula step for step.
6. **Linear attention (KDA).** Fused `qkv_proj` / `fbg_a_proj` / single depthwise
   `qkv_conv`; gating through `compute_g_safe` with `linear_lower_bound` = HF
   `ForgetGate` safe path. The fused gated-delta kernel is skipped for head_dim < 32
   (upstream guard), so the tiny config uses the sequential ops path.
7. Cache layout: `CacheList(latent KVCache, indexer KVCache, PoolingCache, projected KVCache)`
   per DSA layer, `ArraysCache(2)` per linear layer; HF keeps everything in one
   `DynamicCache` layer object.

### qwen4_exp
1. **Decode entry point.** `LanguageModel.__call__` routes single-token decode with a
   cache through `Qwen4ExpBatchInvariantForward`, which re-implements the layer without
   calling `Qwen4ExpAttention.__call__`; the substituted attention would be bypassed.
   The adapter therefore drives `Qwen4ExpModel` directly with explicit 2-D positions
   `offset + arange(L)` and applies `lm_head` itself, for prefill and decode alike.
   Adapter full prefill equals the public forward to 4.5e-8; the adapter's incremental
   decode equals its own prefill to 4.5e-8 (fp32 kernels).
2. **QSA.** HF: per-(batch, query) python loop producing selected token indices, then a
   token mask added to the causal mask. MLX: vectorized `indexer.select` (compressed
   block keys are cached as `index_block_keys`, absent in HF) + `dispatch_qsa_attention`
   (indexed Metal kernel for bf16/fp16; fp32 falls back to a boolean mask from
   `build_mask` + fused SDPA). Same block pooling (mean → `k_layernorm` → rotary at
   block start), same `use_sparse` rule (`complete_blocks > block_topk`) and same tail.
   The foreign path reuses `indexer.select` → `build_mask` (or a causal mask when the
   indexer returns `None` for short single-token decode) and a dense float32 softmax.
3. **Rotary.** Fused Metal M-RoPE kernel (interleaved style), 2-D text positions; the
   adapter rephases foreign keys with the *same* `rotary_emb.apply_rotary` at recency
   positions (negative positions are fine).
4. **Gated DeltaNet.** `_normalize_qk` = HF `l2norm` (eps inside the sqrt) + `D^-0.5`
   on q; prefill uses the chunked scan (`gated_delta_chunked`) or the sequential ops,
   decode the ops; HF uses `torch_chunk_gated_delta_rule` for prefill and the recurrent
   rule for decode. Equal to 3e-8 on the tiny config.
5. Gated residual (hyper-connection), MoE router (softmax → top-k → renormalize), shared
   expert gate, `(1+w)` RMSNorm: same formulas, `mx.compile`d.
6. Cache layout: `QSAKVCache` (rotated keys, values, `index_keys`, `index_position_ids`,
   `index_block_keys`, `index_block_ratio`) per attention layer, `ArraysCache(2)` per
   linear layer.

## 5. Precision findings (the important one)

**mlx 0.32.2 on Apple M5 (`applegpu_g17`) runs fp32 GEMM and fused SDPA at TF32-class
precision.** Measured in-process (`mlx_fixtures.gemm_precision`, `[16,64]@[64,32]`,
normalized max error):

| kernel / setting | vs fp64 | vs fp64 with operands rounded to 10-bit mantissa |
|---|---|---|
| Metal, default arch (`applegpu_g17s`), M ≥ 2 rows | 7.7e-4 | **1.3e-7** |
| Metal, default arch, M = 1 (GEMV) | 5e-8 | — |
| Metal, `MLX_METAL_GPU_ARCH=applegpu_g16s` (or g13/g14/g15) | 2.7e-7 | — |
| CPU stream | 2.7e-7 | — |
| Metal fused SDPA, default arch | 1.1e-3 | — |
| Metal fused SDPA, `applegpu_g16s` | 1.0e-7 | — |

The fp32 product matches the fp64 product of TF32-rounded inputs to 1.3e-7, i.e. the
M5 kernel selection rounds operands to TF32 (bf16 and fp16 rounding do not fit: 2.3e-3
and 8.7e-4). Elementwise ops, `mx.fast.rms_norm`, softmax, conv1d, cumsum and
`mx.compile`d kernels are all ~1e-7. No public MLX API switches this; the
`MLX_METAL_GPU_ARCH` override (read once when the Metal device is created) selects the
older kernels and restores true fp32.

Consequences, all measured (§1 tables):
- Cross-runtime error is 1.8e-4 (Qwen) / 4.1e-3 (GLM) in TF32 mode against a torch
  fp32-vs-fp64 floor of 3.8e-8 / 7.2e-7; with fp32 kernels the MLX port sits *on* that
  floor (4.5e-8 / 7.2e-7), which is the evidence that the weight and operator mapping is
  exact.
- Prefill (GEMM, TF32) and decode (GEMV, fp32) disagree with each other inside MLX by
  the same amounts; the D19 2e-5 prefill/decode qualification cannot pass in TF32 mode.
  GLM blocks of ≤ 8 tokens are exempt because `linear()` runs them token-wise.
- The test module sets `os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g16s")`
  before the first Metal use and `test_metal_fp32_gemm_is_full_precision` verifies the
  override is in effect in that process (it fails with the measured error if MLX was
  initialised earlier in the session, e.g. by another test module). Tolerances were
  chosen for fp32; nothing else about the adapters depends on the override.
- Real bf16 checkpoints on M5 will see TF32-class matmuls regardless; the preregistered
  lower-precision tolerance for real weights (ADAPTERS.md §6) must account for it.

## 6. Sparse-indexer ties (semantic runtime difference, not precision)

Both indexers score with `relu`, so with random weights several blocks/pools of a
query often score **exactly 0**. When the top-k cut falls inside such a tie, the
selected set depends on the tie-break: `torch.topk` keeps the lower index,
`mx.argpartition` the higher. Observed: Qwen block scores `[0, 0, 1.398]` at query 11
(torch picks blocks {0,2}, MLX {1,2}); GLM pool scores `[0, −4.8e-6, 0, 0.0043]` at
query 17 (torch {2,3}, MLX {0,3}). A flipped selection changes that layer's output at
that position and every later layer, so `mlx_fixtures.selection_ties` reproduces the
upstream scoring on the recorded attention inputs and reports positions whose cut is
tied (`TIE_TOLERANCE=1e-5`). `test_cross_runtime_stock_parity` asserts parity strictly
before the first tie, asserts the tail too, and only `xfail`s when the tail differs
*and* a tie explains it (an unexplained tail mismatch fails). With the suite's seeds
one case is tied and flips (`qwen4_exp`, length 21, layer 1, position 14: 4.2e-3 max
abs on the tail) and one is tied without flipping (`glm5_next`, length 21).

## 7. Test inventory (35)

| test | count | default mode | TF32 mode |
|---|---|---|---|
| `test_metal_fp32_gemm_is_full_precision` | 1 | pass | **fail** (7.7e-4) |
| `test_cross_runtime_stock_parity[n×family]` — MLX public stock forward vs torch stock, full prefill + cached 5-token continuation, tie-aware | 8 | 7 pass, 1 xfail (tie flip) | **8 fail** (1.5e-4 … 4.1e-3 ≫ 1e-4) |
| `test_stock_parity_incremental_and_hard_off` — adapter vs MLX stock forward, prefix+continuation, one-token decode, input-state immutability, exact hard-off (output and every cache array `torch.equal`), hostile foreign changes output with mass ∈ [0,1], digest unchanged | 8 | pass | **6 fail** (prefill/decode 1.8e-4 Qwen ×4, 3e-3 GLM lengths 12 and 21); GLM lengths 6, 7 pass |
| `test_capture_boundary_matches_native_cache` — Qwen: canonical K is pre-rotary and post-norm (rephased through the model's rotary it equals the cache; unrephased it does not), V unrotated; GLM: cache keys == latent, zero-width values | 2 | pass | pass |
| `test_self_entries_as_foreign_are_attended` — own entries through a `Gate(-1.0)` change the output, positive mass and entry mass | 2 | pass | pass |
| `test_rejections` — non-KV layer, no gate/override, override 1.5, int32 ids, empty ids | 2 | pass | pass |
| `test_identity_replacement_and_state_snapshot[n×family]` — `import_self_prefix` continues exactly like ordinary decoding, rebuilt cache == native cache, snapshot/restore lossless (`torch.equal` on output and every array), corrupted snapshot / incomplete or shifted prefix refused | 6 | pass | pass |
| `test_embeddings_path_matches_ids` | 2 | pass | pass |
| `test_cross_runtime_foreign_path_parity[override×family]` — same canonical entries (torch capture; MLX capture compared to it at 2e-6) fed to both adapters with override 1.0 and 0.5: output, foreign mass and entry mass within 2e-5, also on a cached continuation | 4 | pass | **4 fail** |
| **total** | **35** | **34 passed, 1 xfailed** | **19 failed, 16 passed** |

Not covered by the D19 suite lengths 127 / 256 / deployment length (the torch suite
does not run them either on tiny configs); qualification on real weights is M4.2+.

## 8. MLX limitations and decisions recorded

1. **Lazy evaluation / `mx.eval` placement.** Cache objects hold lazy graphs after a
   forward. `MlxHybridAdapter` forces evaluation of the output and of every array
   reachable from the cache list (`eval_state`) before returning, before `deepcopy`,
   before snapshotting and after identity import. Without this a deep copy of a state
   drags the whole unevaluated graph along and a snapshot may evaluate the same graph
   twice. `np.array(mx_array)` at the torch boundary also forces evaluation.
2. **Deep copy of caches** (`copy.deepcopy` of `ArraysCache`, `QSAKVCache`, `CacheList`
   of `KVCache`/`PoolingCache`) works on mlx 0.32.2 and is used for input-state
   immutability, exactly like the torch base.
3. **Quantized caches** (`QuantizedKVCache`, `QSAQuantizedKVCache`, TurboQuant) and
   **batched / left-padded caches** (`BatchQSAKVCache`, `ArraysCache.left_padding`) are
   refused explicitly (`require_plain_cache`) in the foreign path and in
   `import_self_prefix`. The dense foreign path needs float `[B,H,T,D]` keys and the
   identity import writes float keys/values; supporting quantized caches means
   dequantize → append → requantize and is not designed yet.
4. **Gated-delta Metal kernel and head_dim 8.** `qwen3_5/gated_delta.py`'s fused kernel
   is templated on `head_dim/32` and fails to *compile* for the tiny config
   ("zero-length arrays are not permitted in C++"). Kernel selection is
   `use_kernel=not self.training`, so the fixture flags the two `Qwen4ExpGatedDeltaNet`
   *instances* (not children) as training; nothing else reads the flag. The adapter
   consequently never calls `model.eval()` (it requires an eval-mode model and refuses
   otherwise) because a recursive `eval()` would reset the flag. Real checkpoints
   (head_dim 128) do not need this. GLM's `models/gated_delta.py` already guards
   `head_dim < 32`.
5. **Batch-invariant decode bypass** (§4 Qwen 1): the adapter uses `Qwen4ExpModel`
   directly. The public `LanguageModel.__call__` is still what the cross-runtime stock
   parity compares against.
6. **`LanguageModel` needs a VLM config** for text-only rope indexing (§2 shim).
7. **No autograd across the boundary.** Gate logits are consumed as
   `float(logsigmoid(logit))`; MLX cannot return gradients to a torch `Gate`. Gate
   training against an MLX receiver would need the gate applied on the torch side to
   the returned masses, or an MLX-native gate. Same for the mailbox marker
   (`forward_embeddings` exists but has no backward).
8. **Stock-path bit-exactness.** With no foreign entries (or override 0) the exact
   upstream `__call__` / `_attend` runs; capture adds one `k_norm` recompute (Qwen) or a
   reference to `new_latent` (GLM) and records the attention input. Hard-off is verified
   with `torch.equal` on the output and every cache array.
9. **Foreign path is dense and float32** (`dense_gated_attention`): QSA/DSA selection →
   boolean mask; foreign columns carry `log g`; softmax `precise=True`; masses from the
   same weights. It matches the torch adapters to ≤ 2e-5 (fp32 kernels).
10. **Tie-breaks** (§6) are a runtime property; neither side is wrong.
11. **TF32 GEMM on M5** (§5) — the single biggest MLX finding of this stage.
12. **`mlx_fixtures.build_family` imports the tiny configs from `tests/test_adapters_next.py`**
    (the task asked for the exact same configs); the module is test support, not runtime.

## 9. Follow-ups

- Hoist `MlxHybridAdapter`, `dense_gated_attention`, `require_plain_cache` into
  `telepathy/adapters/mlx_base.py`; register `qwen4_exp/mlx`, `glm5_next/mlx` in the
  adapter registry with the qualification evidence above.
- Decide the M5 policy for real weights: run with `MLX_METAL_GPU_ARCH=applegpu_g16s`
  (fp32 kernels, slower) or preregister a TF32-aware tolerance.
- Real-checkpoint qualification (bf16, head_dim 128/256, quantized caches, lengths
  127/256/deployment) on oMLX's vendored code.
