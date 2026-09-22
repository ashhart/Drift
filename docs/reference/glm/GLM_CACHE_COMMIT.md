# GLM cache commit verification

The live receiver previously acknowledged the publication file digest after
issuing cache writes and synchronizing the device, without comparing the
destination bytes. Synthetic faults at the real connector `wait_for_save`
callsite reproduced acknowledged dropped writes, acknowledged corrupted writes,
and aliased destination slots accepted before layer mutation. These tests expose
an unchecked receipt assumption; they do not establish that the hardware dropped
historical writes or explain the recorded reverse recall misses.

`glm_cache_commit.py` validates destination shape, integer dtype, device, bounds
and uniqueness while every layer is staged. All layers must pass before any write.
It then writes only the packed 528-byte MLA latent portion and compares every
destination byte with the staged quantized publication before the receiver moves
its cursor or emits an applied receipt. The RoPE tail, other cache slots and
non-target layers remain unchanged. A mismatch or exception poisons the session,
including after a partial write, and blocks retries and subsequent taps. Error
markers and logs retain fixed failure text and stack locations, not tensor values.

The receipt still identifies the source file and row count; it does not establish
attention selection, correct interpretation, or recovery of a private fact.
Verification adds cache reads and reductions; native CUDA cost is unmeasured.
The flat connector deployment must include `glm_cache_commit.py`.

## Executed regression scope

The baseline receiver and connector tests passed 26 tests. The new actual-callsite
regression initially failed all three cases in `tests/test_glm_cache_commit.py`.
After the fix these passed, along with additional malformed-index, poisoned-retry,
native-slot and RoPE-preservation checks. Tests use synthetic CPU Torch caches,
not a stand-in semantic model, and load no checkpoint.

Run the focused tests with the project's existing Python environment:

```bash
python -m pytest -q tests/test_glm_cache_commit.py tests/test_live_receivers.py tests/test_vllm_glm53_inject.py
```

Native CUDA execution, quantified overhead and causal recall recovery remain
BLOCKED pending an authorized native diagnostic; CPU success is not their pass.

## Native selection remains unqualified

Read-only inspection of the deployed vLLM source on 2026-09-21 found:

| Source | SHA-256 |
| --- | --- |
| `vllm/model_executor/models/deepseek_v2.py` | `58d8916458de7c6f73b40bfef9d2f57bdd6fa0fb79be9e269331af6e66149fe2` |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | `22d1d98bd475b1dc22de85d0e414d0490860de50c8bbd26371175e5ea6411116` |

`Indexer` computes selector keys from native hidden states using its distinct
`wk_weights_proj`, key normalization and rotary operation. The current reverse
payload provides MLA latents, not those hidden states or a qualified learned
selector-key mapping. Writing arbitrary selector keys, zeroing them, or deriving
them by reshaping the MLA latent would change the intervention without evidence.

The native `topk_indices_buffer` is reused. Dense-prefill routing can deliberately
leave it untouched because the main attention path does not consume it. Reading
this buffer later in the connector cannot establish per-layer, per-query foreign
slot coverage and could report stale indices as current evidence.

A valid numeric-only diagnostic must observe the actual attention consumer before
buffer reuse, bind rows to the current request and committed foreign span, record
whether attention is dense or sparse, and emit only aggregate selected-slot
counts. No such deployed observation exists in the recorded runs, and no native
selector hook, synthetic coverage claim, weight change or model run was added.
