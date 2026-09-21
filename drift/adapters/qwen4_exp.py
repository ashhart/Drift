"""Qwen3.8-Flash-Next (`qwen4_exp`) adapter on transformers 5.17.0.

KV-bearing layers are the `full_attention` entries of `layer_types`. Canonical K is
captured after `k_norm` and before rotary; V is captured unrotated. Foreign keys are
rephased with the model's own M-RoPE at D16 recency positions (partial rotary, first
`head_dim * partial_rotary_factor` dims) and enter the same softmax as native keys
with `log g` added; they bypass the QSA indexer (research note §4).
"""
from __future__ import annotations
from typing import Any
import torch
from drift.core.types import KV
from .base import Descriptor, ForeignEntries, HybridAdapter, append_foreign_columns

REQUIRED_VERSION = "5.17.0"


def _stock():
    import transformers
    if transformers.__version__ != REQUIRED_VERSION:
        raise RuntimeError(f"qwen4_exp adapter is qualified on transformers=={REQUIRED_VERSION}")
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    return m


class DriftQwen4ExpAttention:
    """Mixin forward; bound onto the stock attention class at substitution time."""

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        m = _stock()
        ctx, rotary = self._drift_ctx, self._drift_rotary
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask
        position_embeddings = (x[:, -hidden_states.shape[1]:, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Canonical boundary: after native key normalization, before rotary; V unrotated. [T,Hkv,D]
        ctx.capture[self.layer_idx] = KV(key_states[0].transpose(0, 1), value_states[0].transpose(0, 1))

        cos, sin = position_embeddings
        query_states, key_states = m.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        native_tokens = key_states.shape[2]
        log_prior = ctx.log_prior(self.layer_idx)
        if log_prior is not None:
            entries = ctx.foreign.layers[self.layer_idx]
            if not isinstance(entries, KV):
                raise ValueError("qwen4_exp receives kv_split foreign entries")
            entries.check()
            fk = entries.k.to(key_states).transpose(0, 1)[None]     # [1,Hkv,Tf,D]
            fv = entries.v.to(value_states).transpose(0, 1)[None]
            virtual = ctx.virtual_positions(hidden_states.device)
            # Text-only M-RoPE: the three position streams coincide.
            fcos, fsin = rotary(hidden_states, virtual.view(1, 1, -1).expand(3, 1, -1))
            fk = m.apply_rotary_pos_emb(fk, cos=fcos, sin=fsin)
            key_states = torch.cat((key_states, fk), dim=2)
            value_states = torch.cat((value_states, fv), dim=2)
            attention_mask = append_foreign_columns(attention_mask, fk.shape[2], log_prior)

        attn_output, attn_weights = m.eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs)
        if log_prior is not None:
            foreign_weights = attn_weights[0, :, :, native_tokens:]              # [Hq, Q, Tf]
            ctx.mass[self.layer_idx] = foreign_weights.sum(-1).transpose(0, 1)
            ctx.entry_mass[self.layer_idx] = foreign_weights.mean(dim=(0, 1))

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen4ExpAdapter(HybridAdapter):
    @classmethod
    def from_local(cls, path: str, device: str = "cpu", dtype=None) -> "Qwen4ExpAdapter":
        """Load a local checkpoint directory (no downloads) with eager attention."""
        import torch as _torch
        import transformers
        _stock()
        kwargs = {"local_files_only": True, "trust_remote_code": False, "attn_implementation": "eager"}
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            model = transformers.Qwen4ExpForCausalLM.from_pretrained(path, **kwargs)
        except Exception:
            model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model.config._attn_implementation = "eager"
        return cls(model.to(device).eval())
    def _substitute(self) -> None:
        m = _stock()
        model = self.model
        text = model.model if hasattr(model.model, "layers") else model.model.language_model
        cfg = text.config
        if cfg.model_type != "qwen4_exp_text":
            raise ValueError("expected a qwen4_exp text model")
        # The config normalizes checkpoint "full_attention" entries to "qwen_sparse_attention".
        if any(t not in {"linear_attention", "qwen_sparse_attention"} for t in cfg.layer_types):
            raise ValueError("unknown layer type; requalify the adapter")
        kv_layers = tuple(i for i, t in enumerate(cfg.layer_types) if t == "qwen_sparse_attention")
        rotary = text.rotary_emb
        patched = type("DriftQwen4ExpTextAttention", (DriftQwen4ExpAttention, m.Qwen4ExpTextAttention), {})
        for i in kv_layers:
            attention = text.layers[i].self_attn
            if type(attention) is not m.Qwen4ExpTextAttention:
                raise ValueError("unexpected attention class; requalify the adapter")
            attention.__class__ = patched
            attention._drift_ctx = self.ctx
            attention._drift_rotary = rotary
        self._text = text
        self.descriptor = Descriptor("qwen4_exp", kv_layers, "kv_split", "partial_rope",
                                     cfg.num_key_value_heads, cfg.head_dim)

    def native_kv_from_canonical(self, layer, entry, positions):
        m = _stock()
        if not isinstance(entry, KV):
            raise ValueError("qwen4_exp canonical entries are KV")
        entry.check()
        k = entry.k.to(self.device).transpose(0, 1)[None]
        cos, sin = self._text.rotary_emb(k, positions.view(1, 1, -1).expand(3, 1, -1))
        return m.apply_rotary_pos_emb(k, cos=cos, sin=sin), entry.v.to(self.device).transpose(0, 1)[None].clone()

    def _bind(self) -> None:
        for i in self.descriptor.kv_layers:
            self._text.layers[i].self_attn._drift_ctx = self.ctx

    def _run(self, ids: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        self._bind()
        result = self.model(input_ids=ids[None], past_key_values=state, use_cache=True)
        output = result.logits if hasattr(result, "logits") else result.last_hidden_state
        return output[0], result.past_key_values

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self._text.embed_tokens(ids.to(self.device))

    def _run_embeds(self, embeddings, state, ple_ids):
        self._bind()
        kwargs = {}
        if self._text.config.ple_layer_ids:
            if ple_ids is None:
                raise ValueError("this checkpoint uses PLE; supply ple_ids for every embedded row")
            kwargs["ple_input_ids"] = ple_ids.to(self.device)[None]
        result = self.model(inputs_embeds=embeddings[None], past_key_values=state, use_cache=True, **kwargs)
        output = result.logits if hasattr(result, "logits") else result.last_hidden_state
        return output[0], result.past_key_values
