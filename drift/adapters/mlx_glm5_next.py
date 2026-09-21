"""GLM-5.3-Flash (`glm5_next`) adapter on the MLX runtime (mlx-vlm 0.7.1).

Same contract and canonical boundary as `glm5_next.py`: the normalized MLA latent
`kv_a_layernorm(kv_a_proj_with_mqa(x))`, `[T, kv_lora_rank]`. NoPE, so foreign latents
are not rephased. They enter the same softmax as native latents with `log g` added and
bypass the DSA indexer.

Operator-boundary difference from HF (docs/research/M4_MLX_NOTES.md): mlx-vlm absorbs
`kv_b_proj` into the attention (`embed_q` maps queries into latent space, `unembed_out`
maps the latent-weighted output back to `v_head_dim`) and caches the latent itself, so
the MLX native cache *is* the canonical entry. Per-head K/V are never materialized;
`native_kv_from_canonical` therefore returns the latent with a zero-width value block.
The foreign block is computed in latent space with the same absorbed weights, which is
algebraically the HF `expand_kv` path.
"""
from __future__ import annotations
import numpy as np
import torch
from drift.core.types import KV
from .base import Descriptor
from .mlx_qwen4_exp import MlxHybridAdapter, _mx, dense_gated_attention, require_plain_cache, to_mx, to_torch

REQUIRED_MLX_VLM = "0.7.1"


def _stock():
    import mlx_vlm
    if mlx_vlm.__version__ != REQUIRED_MLX_VLM:
        raise RuntimeError(f"MLX glm5_next adapter is qualified on mlx-vlm=={REQUIRED_MLX_VLM}")
    from mlx_vlm.models.glm5_next import language as m
    return m


def topk_to_allowed(topk, key_length: int):
    """[B,L,K] DSA indices (-1 invalid) -> [B,1,L,T] boolean visibility (HF build_attention_mask_from_topk)."""
    mx = _mx()
    b, l, _ = topk.shape
    valid = (topk >= 0) & (topk < key_length)
    scatter = mx.where(valid, topk, key_length)
    mask = mx.put_along_axis(mx.zeros((b, l, key_length + 1), dtype=mx.bool_), scatter, mx.array(True), axis=-1)
    return mask[..., :key_length][:, None]


class DriftMlxGlm5NextAttention:
    """Mixin; bound onto `Glm5NextAttention` at substitution time."""

    def __call__(self, x, padding_mask=None, cache=None, prev_topk_indices=None, last_only=False):
        self._drift_last_input = x
        if self._drift_ctx.log_prior(self._drift_layer) is not None:
            if padding_mask is not None:
                raise ValueError("foreign attention is qualified for unpadded batches of one")
            require_plain_cache(cache)
        return super().__call__(x, padding_mask, cache, prev_topk_indices, last_only)

    def _attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
        mx = _mx()
        ctx, layer = self._drift_ctx, self._drift_layer
        # Canonical boundary: the normalized latent of the new tokens, [T, C].
        ctx.capture[layer] = new_latent[0, 0]
        log_prior = ctx.log_prior(layer)
        if log_prior is None:
            return super()._attend(q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
        if last_only:
            raise ValueError("foreign attention returns every query row")
        foreign = ctx.foreign.layers[layer]
        if not isinstance(foreign, torch.Tensor) or foreign.ndim != 2 or foreign.shape[1] != self.kv_lora_rank:
            raise ValueError("glm5_next receives mla_latent foreign entries [T, kv_lora_rank]")
        if not torch.isfinite(foreign).all():
            raise ValueError("nonfinite foreign latent")
        b, _, l, _ = q.shape
        native_tokens = latent.shape[2]
        allowed = topk_to_allowed(topk, native_tokens)
        query = self.embed_q(q)                                                 # [B,H,L,C]: q W_k^T per head
        f = to_mx(foreign)[None, None].astype(latent.dtype)                     # [1,1,Tf,C], NoPE: no rephase
        output, foreign_weights = dense_gated_attention(query, latent, latent, allowed, f, f, log_prior, self.scale)
        output = self.unembed_out(output)                                       # [B,H,L,v]
        ctx.mass[layer] = foreign_weights[0].sum(-1).T                          # [L, H]
        ctx.entry_mass[layer] = foreign_weights[0].mean(axis=(0, 1))            # [Tf]
        return output.transpose(0, 2, 1, 3).reshape(b, l, -1), topk


class MlxGlm5NextAdapter(MlxHybridAdapter):
    def _substitute(self) -> None:
        m = _stock()
        lm = self.model
        cfg = lm.config
        if cfg.model_type != "glm5_next_text":
            raise ValueError("expected a glm5_next text model")
        if any(t not in {"linear_attention", "deepseek_sparse_attention"} for t in cfg.layer_types):
            raise ValueError("unknown layer type; requalify the adapter")
        if cfg.qk_rope_head_dim != 0:
            raise ValueError("adapter is qualified for the NoPE configuration only")
        kv_layers = tuple(i for i, t in enumerate(cfg.layer_types) if t == "deepseek_sparse_attention")
        patched = type("DriftMlxGlm5NextAttentionImpl", (DriftMlxGlm5NextAttention, m.Glm5NextAttention), {})
        for i in kv_layers:
            attention = lm.model.layers[i].self_attn
            if type(attention) is not m.Glm5NextAttention:
                raise ValueError("unexpected attention class; requalify the adapter")
            attention.__class__ = patched
            attention._drift_ctx = self.ctx
            attention._drift_layer = i
        self._text = lm.model
        self.descriptor = Descriptor("glm5_next", kv_layers, "mla_latent", "none",
                                     cfg.num_attention_heads, cfg.kv_lora_rank)

    def _bind(self) -> None:
        for i in self.descriptor.kv_layers:
            self._text.layers[i].self_attn._drift_ctx = self.ctx

    def _kv_cache(self, state, layer):
        return state[layer][0]              # CacheList(latent KVCache, index KVCache, PoolingCache, projected KVCache)

    def _run(self, inputs, cache, inputs_embeds=None):
        hidden = self._text(inputs, inputs_embeds=inputs_embeds, cache=cache)
        return hidden[0]                    # [T, hidden]: the torch fixture is headless too

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        mx = _mx()
        out = self._text.embed_tokens(mx.array(ids.detach().cpu().numpy().astype(np.int32)))
        mx.eval(out)
        return to_torch(out)

    def native_kv_from_canonical(self, layer, entry, positions):
        mx = _mx()
        if not isinstance(entry, torch.Tensor) or entry.ndim != 2 or entry.shape[1] != self.descriptor.head_dim:
            raise ValueError("glm5_next canonical entries are [T,C] latents")
        latent = to_mx(entry)[None, None]                                       # [1,1,T,C]; the cache stores the latent
        return latent, mx.zeros((1, 1, latent.shape[2], 0), dtype=latent.dtype)
