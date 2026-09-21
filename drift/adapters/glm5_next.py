"""GLM-5.3-Flash (`glm5_next`) adapter on transformers 5.17.0.

KV-bearing layers are the `deepseek_sparse_attention` entries of `layer_types`. The
canonical entry is the normalized MLA latent `kv_a_layernorm(kv_a_proj_with_mqa(x))`,
shape [T, kv_lora_rank]; per-head K and V are derived by the receiver's own frozen
`kv_b_proj`. The model is NoPE (`qk_rope_head_dim == 0`), so foreign entries are not
rephased. Foreign latents are expanded with `expand_kv` and enter the same softmax
as native keys with `log g` added; they bypass the DSA indexer.
"""
from __future__ import annotations
from typing import Any
import torch
from .base import Descriptor, HybridAdapter, append_foreign_columns

REQUIRED_VERSION = "5.17.0"


def _stock():
    import transformers
    if transformers.__version__ != REQUIRED_VERSION:
        raise RuntimeError(f"glm5_next adapter is qualified on transformers=={REQUIRED_VERSION}")
    from transformers.models.glm5_next import modeling_glm5_next as m
    return m


class DriftGlm5NextAttention:
    def forward(self, hidden_states, attention_mask, past_key_values=None, prev_topk_indices=None, **kwargs):
        m = _stock()
        ctx = self._drift_ctx
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Canonical boundary: the normalized latent, [T, C]. Heads are derived by kv_b_proj.
        ctx.capture[self.layer_idx] = k_pass[0, 0]

        key_states, value_states = self.expand_kv(k_pass, k_rot)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.indexer is not None:
            topk_indices = self.indexer(hidden_states=hidden_states, q_resid=q_resid,
                                        attention_mask=attention_mask, past_key_values=past_key_values)
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        native_tokens = key_states.shape[2]
        attention_mask = self.build_attention_mask_from_topk(
            topk_indices=topk_indices, query_states=query_states, kv_length=native_tokens)

        log_prior = ctx.log_prior(self.layer_idx)
        if log_prior is not None:
            latent = ctx.foreign.layers[self.layer_idx]
            if not isinstance(latent, torch.Tensor) or latent.ndim != 2 or latent.shape[1] != self.kv_lora_rank:
                raise ValueError("glm5_next receives mla_latent foreign entries [T, kv_lora_rank]")
            if not torch.isfinite(latent).all():
                raise ValueError("nonfinite foreign latent")
            latent = latent.to(key_states)[None, None]                       # [1,1,Tf,C]
            empty_rot = latent.new_empty(1, 1, latent.shape[2], self.qk_rope_head_dim)
            fk, fv = self.expand_kv(latent, empty_rot)                        # NoPE: no rephase
            key_states = torch.cat((key_states, fk), dim=2)
            value_states = torch.cat((value_states, fv), dim=2)
            attention_mask = append_foreign_columns(attention_mask, fk.shape[2], log_prior)

        attn_output, attn_weights = m.eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs)
        if log_prior is not None:
            foreign_weights = attn_weights[0, :, :, native_tokens:]
            ctx.mass[self.layer_idx] = foreign_weights.sum(-1).transpose(0, 1)
            ctx.entry_mass[self.layer_idx] = foreign_weights.mean(dim=(0, 1))

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None


class Glm5NextAdapter(HybridAdapter):
    @classmethod
    def from_local(cls, path: str, device: str = "cpu", dtype=None) -> "Glm5NextAdapter":
        """Load a local text checkpoint directory (no downloads) with eager attention."""
        m = _stock()
        kwargs = {"local_files_only": True, "trust_remote_code": False, "attn_implementation": "eager"}
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = m.Glm5NextTextModel.from_pretrained(path, **kwargs)
        model.config._attn_implementation = "eager"
        return cls(model.to(device).eval())
    def _substitute(self) -> None:
        m = _stock()
        text = self.model
        cfg = text.config
        if cfg.model_type != "glm5_next_text":
            raise ValueError("expected a glm5_next text model")
        if any(t not in {"linear_attention", "deepseek_sparse_attention"} for t in cfg.layer_types):
            raise ValueError("unknown layer type; requalify the adapter")
        if cfg.qk_rope_head_dim != 0:
            raise ValueError("adapter is qualified for the NoPE configuration only")
        kv_layers = tuple(i for i, t in enumerate(cfg.layer_types) if t == "deepseek_sparse_attention")
        patched = type("DriftGlm5NextTextAttention", (DriftGlm5NextAttention, m.Glm5NextTextAttention), {})
        for i in kv_layers:
            attention = text.layers[i].self_attn
            if type(attention) is not m.Glm5NextTextAttention:
                raise ValueError("unexpected attention class; requalify the adapter")
            attention.__class__ = patched
            attention._drift_ctx = self.ctx
        self._text = text
        self.descriptor = Descriptor("glm5_next", kv_layers, "mla_latent", "none",
                                     cfg.num_attention_heads, cfg.kv_lora_rank)

    def native_kv_from_canonical(self, layer, entry, positions):
        if not isinstance(entry, torch.Tensor) or entry.ndim != 2:
            raise ValueError("glm5_next canonical entries are [T,C] latents")
        attention = self._text.layers[layer].self_attn
        latent = entry.to(self.device)[None, None]
        k, v = attention.expand_kv(latent, latent.new_empty(1, 1, latent.shape[2], attention.qk_rope_head_dim))
        return k, v

    def _bind(self) -> None:
        for i in self.descriptor.kv_layers:
            self._text.layers[i].self_attn._drift_ctx = self.ctx

    def _run(self, ids: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        self._bind()
        result = self.model(input_ids=ids[None], past_key_values=state, use_cache=True)
        return result.last_hidden_state[0], result.past_key_values

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self._text.embed_tokens(ids.to(self.device))

    def _run_embeds(self, embeddings, state, ple_ids):
        self._bind()
        result = self.model(inputs_embeds=embeddings[None], past_key_values=state, use_cache=True)
        return result.last_hidden_state[0], result.past_key_values
