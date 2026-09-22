# Serving-stack integration: vLLM, SGLang, oMLX

**Status:** owner decision OD5 (19 September 2026): Drift must work with the serving
stacks people actually run. This overrides P1 §7 ("do not use paged caches") and D15's
"dedicated worker only" stance. The dedicated adapters stay as the correctness reference
and the fully-featured research path; serving integrations are qualified against them.

## What was inspected (read-only, spark-a.invalid `glm53-exl3-head` container)

- vLLM `0.1.dev20051+g487ecf187` (custom build, EXL3 weights, TP across two Sparks, MTP
  speculative decoding, prefix caching on, `--enforce-eager`).
- The **KV connector v1 API** is present (`vllm/distributed/kv_transfer/kv_connector/v1/base.py`)
  with in-tree connectors (example, LMCache, NIXL, Mooncake, offloading, multi) and hybrid
  support (`SupportsHMA`, `ssm_conv_transfer_utils.py` for Mamba/GDN conv and temporal state).
  Out-of-tree connectors load through `--kv-transfer-config` with a module path.
- Attention backends include sparse MLA for this GPU (`flashinfer_mla_sparse_sm120.py`), an
  indexer backend with its own cache, GDN/linear attention, and `flex_attention.py`.
- Qwen3.8 has its own vLLM image and unit (`qwen38-vllm.service`, conflicts with the GLM unit).

## The seam: a Drift KV connector

The connector API does two things, and both map onto Drift:

| Connector hook | vLLM meaning | Drift use |
|---|---|---|
| `save_kv_layer(layer, kv, attn_metadata)` | export a layer's KV for a request's tokens | **AttentionTap**: read native cache rows per KV-bearing layer, de-rotate to canonical where the model uses RoPE, hand them to the member's writer |
| `get_num_new_matched_tokens` + `update_state_after_alloc` + `start_load_kv` | "the first N prompt tokens already exist externally; load them instead of computing" | **Foreign memory injection**: the receiver's prompt starts with N reserved placeholder tokens; the connector claims them and writes translated foreign entries into their cache slots |

### Connector mode semantics (what a stock serving stack can do without kernel changes)

- Foreign entries occupy **real cache slots at positions 0..N−1** of the receiving request.
  For RoPE models the connector rotates translated canonical K at those positions; GLM is
  NoPE so nothing is rotated. D16 recency positions become "prefix positions": foreign
  memory sits before the prompt, newest last.
- **Hybrid models:** linear-attention layers (Qwen GatedDeltaNet, GLM KDA) get the state "as
  if nothing was read" for the placeholder span (the initial state), supplied through the
  hybrid-state path. Those layers never see foreign content, which matches the dedicated
  adapters, where only KV-bearing layers attend to foreign entries.
- **Sparse indexers:** the indexer cache is another registered cache. First experiments keep
  total context ≤ the indexer budget (2,048) so every slot is selected. Beyond that the reader
  needs an indexer-key head (a declared, trained sidecar).
- **Exact hard-off** is "no placeholders, no injection": byte-for-byte the stock request.

### What connector mode loses versus the dedicated adapters

1. **No gate inside the softmax.** Fused paged kernels take no per-key bias, so there is no
   `log g` prior and no learned scalar gate. Salience must be carried by the translated keys
   (behavior training), and control is on/off plus how many entries are injected.
2. **No per-entry attention mass** from the kernel, so mailbox "attended" tracking is not
   available in this mode; incorporation is still testable by counterfactual replay (inject vs
   do not inject).
3. **Epoch granularity is a request**, not a decode step. Live per-step coupling needs either
   chunked requests or the attention-hook mode below.
4. **Cache precision is the server's.** This container uses an FP8 MLA cache; tapped latents
   are dequantized FP8, and injected latents are quantized on write.
5. Interactions to qualify, not assume: speculative decoding (MTP), prefix caching (placeholder
   prefixes must not be shared across runs with different foreign content: salt the placeholder
   block hash per session), CUDA graphs, tensor parallel (KV heads are sharded across ranks).

### Attention-hook mode (later, optional)

Full Drift semantics inside vLLM need a custom attention backend or plugin: a separate
gated foreign block in the softmax. `flex_attention.py` exists and accepts score modifiers, so
a correctness-first implementation is possible; it will be slow and is not needed for E1/E2.

## oMLX (Studio)

oMLX is Python and in-process: the MLX adapters' attention subclasses can be installed as an
oMLX patch (same mechanism as its `mlx_vlm_*_compat` patches), giving full semantics (gate,
mass, per-step coupling). The work is wiring: session contract, bank access from the engine's
scheduler, and publication of tapped rows.

## SGLang

Same idea through its KV-transfer / HiCache storage backends. Not inspected yet.

## Qualification ladder for a serving integration

1. **Connector identity (M0 analogue):** save a request's KV through the connector, serve the
   same prompt again with its KV loaded from the store: outputs must match the stock run
   (within the server's own run-to-run tolerance, measured first).
2. **Tap parity:** canonical entries tapped from the server == canonical entries from the
   dedicated adapter on the same checkpoint and text (tolerance preregistered; FP8 cache noted).
3. **Injection parity:** inject the same translated entries through the connector and through
   the dedicated adapter with `override=1`; compare outputs. They will differ by the known
   semantic gaps above (prefix positions vs recency positions); the gap is measured and reported.
4. Only then E1/E2 through the server.

## What is needed from the owner

- A maintenance window on the Sparks: restart the GLM service with `--kv-transfer-config`
  pointing at the Drift connector (plugin mounted read-only), MTP and prefix caching off
  for qualification runs.
- A small dense model for developing the connector under vLLM without the 164 GB load
  (download permission), or agreement to develop directly against GLM.
- The oMLX loader for Qwen3.8 if upstream mlx-vlm refuses the checkpoint's MTP tensors.
