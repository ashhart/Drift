from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import torch
from torch import nn
from drift.core.types import KV
from drift.core.position import apply_cos_sin, recency_positions
from drift.core.attention import Gate, attend
from drift.core.memory import ForeignView


@dataclass(frozen=True)
class NativeState:
    length: int
    layers: tuple[KV, ...]           # native K is ALREADY rotated here

    def clone(self, *, detach: bool = True) -> NativeState:
        return NativeState(self.length, tuple(kv.clone(detach=detach) for kv in self.layers))


@dataclass(frozen=True)
class StepOutput:
    logits: torch.Tensor             # [T, receiver_vocab]
    state: NativeState
    canonical_delta: Mapping[int, KV]
    foreign_mass: Mapping[int, torch.Tensor]


class FrozenDecoder:
    """Dedicated reference forward, not a generic provider/completion API.

    Supports the dense full-attention Qwen3/Llama layout and the supplied toy.
    No monkeypatches or native cache pollution. Batch one, unquantized, one device,
    eager attention, fixed-frequency split-half rotary only. HF parity is a gate.
    All modules/parameters are the original frozen checkpoint modules.
    """
    def __init__(self, model: nn.Module):
        cfg = model.config
        if cfg.model_type not in {"qwen3", "llama", "drift_toy"}:
            raise ValueError("unsupported model architecture; implement and verify an adapter")
        if getattr(model, "is_quantized", False):
            raise ValueError("quantized models require a separately validated adapter")
        if getattr(cfg, "pretraining_tp", 1) != 1:
            raise ValueError("tensor-parallel checkpoint forward not supported")
        if getattr(cfg, "use_sliding_window", False) or any(
                x != "full_attention" for x in getattr(cfg, "layer_types", [])):
            raise ValueError("sliding/hybrid layers require a separate mask/cache adapter")
        rope = getattr(cfg, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) not in {"default", "llama3"}:
            raise ValueError("dynamic/other RoPE needs explicit lifetime and parity tests")
        model.eval().requires_grad_(False)
        self.model = model
        self.layers = model.model.layers
        self.qheads, self.kvheads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.qheads
        if self.dim % 2 or self.qheads % self.kvheads:
            raise ValueError("unsupported rotary/GQA shape")
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("materialize the whole model on one device")
        self.device = next(iter(devices))
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def from_local_hf(cls, path: str | Path, device: str = "cpu",
                      dtype: torch.dtype = torch.float32) -> FrozenDecoder:
        import transformers
        if transformers.__version__ != "4.56.2":
            raise RuntimeError("reference HF adapter requires transformers==4.56.2; requalify upgrades")
        if not Path(path).is_dir():
            raise FileNotFoundError("provide an already authorized local checkpoint directory")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="eager")
        return cls(model.to(device))

    def rotate(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.model.model.rotary_emb(k.unsqueeze(0), positions[None])
        return apply_cos_sin(k, cos[0], sin[0])

    def empty(self) -> NativeState:
        return NativeState(0, ())

    def import_self_prefix(self, canonical: Mapping[int, KV],
                           positions: torch.Tensor) -> NativeState:
        """M0 identity replacement only. NOT ordinary cross-family memory append."""
        if set(canonical) != set(range(len(self.layers))):
            raise ValueError("self-handoff requires every layer, not an eight-layer subset")
        n = positions.numel()
        if not torch.equal(positions.cpu(), torch.arange(n)) or n == 0:
            raise ValueError("self-handoff requires a complete prefix starting at position zero")
        result = []
        for i in range(len(self.layers)):
            kv = canonical[i]
            kv.check()
            if kv.k.shape != (n, self.kvheads, self.dim):
                raise ValueError("self-prefix shape mismatch")
            k, v = kv.k.to(device=self.device, dtype=self.dtype), kv.v.to(device=self.device, dtype=self.dtype)
            result.append(KV(self.rotate(k, positions.to(self.device)), v.clone()))
        return NativeState(n, tuple(result))

    def forward(self, ids: torch.Tensor | None, state: NativeState | None = None,
                foreign: ForeignView | None = None,
                gates: Mapping[int, Gate] | None = None,
                override: float | None = None,
                embeddings: torch.Tensor | None = None) -> StepOutput:
        # Do NOT wrap this method in no_grad: frozen backbones must pass gradients
        # from the task loss into injected activations and trainable sidecars.
        if (ids is None) == (embeddings is None):
            raise ValueError("provide exactly one of local token IDs or embeddings")
        state = self.empty() if state is None else state
        if ids is not None:
            if ids.ndim != 1 or ids.numel() == 0 or ids.dtype != torch.long:
                raise ValueError("local token IDs must be nonempty int64 [T]")
            hidden = self.model.model.embed_tokens(ids.to(self.device))
        else:
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                raise ValueError("local embeddings must be [T,hidden]")
            hidden = embeddings.to(device=self.device, dtype=self.dtype)
        n = hidden.shape[0]
        if state.length < 0 or (state.length == 0 and state.layers) or (
                state.length > 0 and len(state.layers) != len(self.layers)):
            raise ValueError("invalid native state")
        if state.length + n > self.model.config.max_position_embeddings:
            raise ValueError("reference context budget exceeded")
        positions = torch.arange(state.length, state.length + n, device=self.device)
        all_positions = torch.arange(state.length + n, device=self.device)
        causal = all_positions[None] <= positions[:, None]
        native, captured, masses = [], {}, {}
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            normed = layer.input_layernorm(hidden)
            query = attention.q_proj(normed).reshape(n, self.qheads, self.dim)
            key = attention.k_proj(normed).reshape(n, self.kvheads, self.dim)
            value = attention.v_proj(normed).reshape(n, self.kvheads, self.dim)
            if hasattr(attention, "q_norm"):
                query = attention.q_norm(query)
            if hasattr(attention, "k_norm"):
                key = attention.k_norm(key)
            captured[index] = KV(key, value)
            qr, kr = self.rotate(query, positions), self.rotate(key, positions)
            if state.length:
                past = state.layers[index]
                if past.k.shape != (state.length, self.kvheads, self.dim):
                    raise ValueError("native cache length/shape mismatch")
                kr = torch.cat((past.k, kr))
                value = torch.cat((past.v, value))
            local = KV(kr, value)
            native.append(local)
            imported = None
            if foreign is not None and index in foreign.layers and override != 0.0:
                memory = foreign.layers[index]
                fk = memory.k.to(device=self.device, dtype=self.dtype)
                fv = memory.v.to(device=self.device, dtype=self.dtype)
                virtual = recency_positions(foreign.positions.to(self.device), state.length)
                imported = KV(self.rotate(fk, virtual), fv)
            selected_gate = None if gates is None else gates.get(index)
            output, mass = attend(qr, local, causal, imported, selected_gate, override,
                                  scale=getattr(attention, "scaling", self.dim**-0.5))
            masses[index] = mass
            hidden = hidden + attention.o_proj(output.reshape(n, self.qheads * self.dim))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = self.model.lm_head(self.model.model.norm(hidden))
        return StepOutput(logits, NativeState(state.length + n, tuple(native)), captured, masses)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 foreign: ForeignView | None = None, override: float = 1.0,
                 eos_id: int | None = None) -> list[int]:
        """Greedy diagnostic generation, local IDs only. Count all local tokens."""
        if max_new_tokens <= 0:
            raise ValueError("positive generation budget required")
        output = self.forward(prompt_ids, foreign=foreign, override=override)
        generated = []
        for step in range(max_new_tokens):
            token = int(output.logits[-1].argmax())
            generated.append(token)
            if token == eos_id or step == max_new_tokens - 1:
                break
            output = self.forward(torch.tensor([token], dtype=torch.long, device=self.device),
                                  output.state, foreign=foreign, override=override)
        return generated
