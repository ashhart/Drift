"""Qwen3.8-Flash-Next (`qwen4_exp`) adapter on the MLX runtime (mlx-vlm 0.7.1).

Same contract as `qwen4_exp.py` (docs/ADAPTERS.md), same canonical boundary: K after
`k_norm` and before rotary, V unrotated, `[T, Hkv, D]`. Foreign entries are rephased
with the model's own M-RoPE at recency positions and enter the same softmax as native
keys with `log g` added; they bypass the QSA indexer.

Runtime notes (details in docs/research/M4_MLX_NOTES.md):
- The adapter drives `Qwen4ExpModel` (the decoder stack) directly with explicit 2-D
  text positions and applies the LM head itself. The public `LanguageModel.__call__`
  reroutes single-token decode through a batch-invariant forward that bypasses the
  attention class, so it cannot host the substituted attention.
- Capture and the foreign block are added by swapping `__class__` on the KV-bearing
  attention instances (as the torch adapters do). With no foreign entries (or override
  0) the stock `__call__` runs unchanged; capture recomputes `k_norm` on the same
  projection (deterministic, so identical to the value the stock path uses).
- With foreign entries the QSA selection is turned into a boolean mask and attention is
  computed densely (float32 softmax) so extra columns can be appended; the fused
  dispatch cannot take them.
- Outputs, canonical entries and masses are torch tensors (numpy at the boundary).
  Gate logits are consumed as floats: no gradient flows from MLX into a torch `Gate`.

The shared `MlxHybridAdapter` base lives here because M4.1 may only add these files;
hoisting it into `mlx_base.py` is a follow-up.
"""
from __future__ import annotations
import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
import torch
from torch.nn import functional as F
from drift.core.attention import Gate
from drift.core.position import recency_positions
from drift.core.types import KV
from .base import Descriptor, ForeignEntries, StepOutput

REQUIRED_MLX_VLM = "0.7.1"


def _mx():
    import mlx.core as mx
    return mx


def _stock():
    import mlx_vlm
    if mlx_vlm.__version__ != REQUIRED_MLX_VLM:
        raise RuntimeError(f"MLX qwen4_exp adapter is qualified on mlx-vlm=={REQUIRED_MLX_VLM}")
    from mlx_vlm.models.qwen4_exp import language as m
    return m


# -- conversions -----------------------------------------------------------------

def to_mx(tensor: torch.Tensor):
    return _mx().array(tensor.detach().cpu().contiguous().numpy())


def to_torch(array) -> torch.Tensor:
    mx = _mx()
    if array.dtype == mx.bfloat16:          # numpy has no bfloat16; real checkpoints produce it
        array = array.astype(mx.float32)
    return torch.from_numpy(np.array(array, copy=True))


def _collect_arrays(value: Any, out: list, seen: set) -> None:
    """Every mx.array reachable through dicts, lists, tuples and object attributes."""
    mx = _mx()
    if isinstance(value, mx.array):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_arrays(v, out, seen)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_arrays(v, out, seen)
    elif hasattr(value, "__dict__") and id(value) not in seen:
        seen.add(id(value))
        for v in vars(value).values():
            _collect_arrays(v, out, seen)


def eval_state(state: Any) -> None:
    """Force every cache array (MLX is lazy; a snapshot or deepcopy must see values)."""
    arrays: list = []
    _collect_arrays(state, arrays, set())
    if arrays:
        _mx().eval(*arrays)


# -- per-forward context -------------------------------------------------------------

@dataclass
class _MlxContext:
    foreign: ForeignEntries | None = None
    gates: Mapping[int, Gate] | None = None
    override: float | None = None
    first_query_position: int = 0
    capture: dict[int, Any] = field(default_factory=dict)      # mx arrays until forward returns
    mass: dict[int, Any] = field(default_factory=dict)
    entry_mass: dict[int, Any] = field(default_factory=dict)

    def log_prior(self, layer: int) -> float | None:
        """None means this layer attends natively only. Same rules as base._Context."""
        if self.foreign is None or layer not in self.foreign.layers or self.override == 0.0:
            return None
        if self.override is not None:
            if not 0.0 < self.override <= 1.0:
                raise ValueError("gate override must be in (0,1]; 0 is the exact native path")
            return math.log(self.override)
        gate = None if self.gates is None else self.gates.get(layer)
        if gate is None:
            raise ValueError("foreign attention requires a gate or explicit override")
        return float(F.logsigmoid(gate.logit.detach()))

    def virtual_positions(self):
        assert self.foreign is not None
        positions = recency_positions(self.foreign.positions.cpu(), self.first_query_position)
        return _mx().array(positions.numpy().astype(np.int32))[None]           # [1, Tf]


def dense_gated_attention(queries, keys, values, allowed, foreign_k, foreign_v, log_prior: float, scale: float):
    """Eager batch-one attention with a gated foreign block in the same softmax.

    queries [B,Hq,L,D]; keys/values [B,Hkv,T,D]; allowed [B,1,L,T] bool; foreign_k/v
    [B,Hkv,Tf,Dk]/[B,Hkv,Tf,Dv]. Returns (output [B,Hq,L,Dv], foreign weights [B,Hq,L,Tf])."""
    mx = _mx()
    rep = queries.shape[1] // keys.shape[1]
    if rep > 1:
        keys, values = mx.repeat(keys, rep, axis=1), mx.repeat(values, rep, axis=1)
        foreign_k, foreign_v = mx.repeat(foreign_k, rep, axis=1), mx.repeat(foreign_v, rep, axis=1)
    q = queries.astype(mx.float32) * scale
    native = q @ keys.astype(mx.float32).swapaxes(-1, -2)
    native = mx.where(allowed, native, mx.array(-mx.inf, dtype=mx.float32))
    foreign = q @ foreign_k.astype(mx.float32).swapaxes(-1, -2) + log_prior
    weights = mx.softmax(mx.concatenate([native, foreign], axis=-1), axis=-1, precise=True)
    t = keys.shape[2]
    output = weights[..., :t] @ values.astype(mx.float32) + weights[..., t:] @ foreign_v.astype(mx.float32)
    return output.astype(queries.dtype), weights[..., t:]


def require_plain_cache(cache) -> None:
    """The adapters are qualified on float, single-row caches only."""
    if cache is None:
        return
    if hasattr(cache, "bits") or any(hasattr(c, "bits") for c in getattr(cache, "caches", ())):
        raise ValueError("quantized KV caches are not supported by the MLX adapters yet")
    if hasattr(cache, "left_padding") and getattr(cache, "left_padding", None) is not None:
        raise ValueError("batched / left-padded caches are not supported by the MLX adapters")


def _causal_allowed(query_length: int, key_length: int):
    """[1,1,L,T] bool: query i (the last L of T keys) sees keys <= its own position."""
    mx = _mx()
    offset = key_length - query_length
    return (mx.arange(key_length)[None, :] <= (offset + mx.arange(query_length))[:, None])[None, None]


# -- shared MLX adapter machinery --------------------------------------------------------

class MlxHybridAdapter:
    """MLX counterpart of `base.HybridAdapter`. Private state is the list of per-layer
    mlx-vlm cache objects from `make_cache()`; `forward` deep-copies it."""
    descriptor: Descriptor

    def __init__(self, model):
        # The model is used as configured: adapters never toggle modes on a serving
        # stack (mlx-vlm's `load` already calls `model.eval()`, and the fixture relies on
        # instance-level flags a recursive `eval()` would reset).
        if model.training:
            raise ValueError("call model.eval() before adapting")
        self.model = model
        self.ctx = _MlxContext()
        self._substitute()

    # -- subclass hooks ------------------------------------------------------------
    def _substitute(self) -> None:
        raise NotImplementedError

    def _run(self, inputs, cache, inputs_embeds=None):
        """Run the decoder on [1,T] ids (or [1,T,hidden] embeddings) with `cache`; [T,X]."""
        raise NotImplementedError

    def _kv_cache(self, state: Any, layer: int):
        """The KVCache object holding layer `layer`'s native K/V."""
        raise NotImplementedError

    def _bind(self) -> None:
        raise NotImplementedError

    def native_kv_from_canonical(self, layer: int, entry, positions: torch.Tensor):
        """Stock-cache K/V (mx arrays [1,H,T,*]) for `layer` from canonical entries at exact positions."""
        raise NotImplementedError

    # -- contract ------------------------------------------------------------------
    def make_cache(self):
        return self.model.make_cache()

    def state_length(self, state: Any) -> int:
        return 0 if state is None else int(self._kv_cache(state, self.descriptor.kv_layers[0]).offset)

    def _prepare(self, state, foreign, gates, override):
        if foreign is not None:
            unknown = set(foreign.layers) - set(self.descriptor.kv_layers)
            if unknown:
                raise ValueError(f"foreign entries target non-KV layers {sorted(unknown)}")
        if state is None:
            state = self.make_cache()
        else:
            eval_state(state)
            state = copy.deepcopy(state)
        self.ctx = _MlxContext(foreign, gates, override, self.state_length(state))
        self._bind()
        return state

    def _finish(self, output, state, ctx) -> StepOutput:
        mx = _mx()
        mx.eval(output)
        eval_state(state)
        canonical = {i: self._canonical_to_torch(v) for i, v in ctx.capture.items()}
        mass = {i: to_torch(m) for i, m in ctx.mass.items()}
        entry_mass = {i: to_torch(m) for i, m in ctx.entry_mass.items()}
        return StepOutput(to_torch(output), state, canonical, mass, entry_mass)

    @staticmethod
    def _canonical_to_torch(value):
        if isinstance(value, tuple):
            return KV(to_torch(value[0]), to_torch(value[1]))
        return to_torch(value)

    def forward(self, ids: torch.Tensor, state: Any = None, foreign: ForeignEntries | None = None,
                gates: Mapping[int, Gate] | None = None, override: float | None = None) -> StepOutput:
        if ids.ndim != 1 or ids.numel() == 0 or ids.dtype != torch.long:
            raise ValueError("local token IDs must be nonempty int64 [T]")
        state = self._prepare(state, foreign, gates, override)
        inputs = _mx().array(ids.detach().cpu().numpy().astype(np.int32))[None]
        try:
            output = self._run(inputs, state)
        finally:
            ctx, self.ctx = self.ctx, _MlxContext()
            self._bind()
        return self._finish(output, state, ctx)

    def forward_embeddings(self, embeddings: torch.Tensor, state: Any = None,
                           foreign: ForeignEntries | None = None, gates: Mapping[int, Gate] | None = None,
                           override: float | None = None, ple_ids: torch.Tensor | None = None) -> StepOutput:
        """Same as `forward` from [T,hidden] embeddings. No autograd path exists into MLX."""
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("local embeddings must be nonempty [T,hidden]")
        state = self._prepare(state, foreign, gates, override)
        inputs_embeds = to_mx(embeddings.detach().float())[None]
        try:
            output = self._run(None, state, inputs_embeds=inputs_embeds)
        finally:
            ctx, self.ctx = self.ctx, _MlxContext()
            self._bind()
        return self._finish(output, state, ctx)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def frozen_digest(self) -> str:
        from mlx.utils import tree_flatten
        digest = hashlib.sha256()
        for name, array in sorted(tree_flatten(self.model.parameters()), key=lambda kv: kv[0]):
            data = np.ascontiguousarray(np.array(array))
            digest.update(name.encode())
            digest.update(str((tuple(data.shape), str(array.dtype))).encode())
            digest.update(data.tobytes())
        return digest.hexdigest()

    # -- private state (the whole cache list, through nested cache objects) ------------------
    @classmethod
    def _flatten(cls, value: Any, path: str, tensors: dict[str, torch.Tensor]) -> Any:
        mx = _mx()
        if isinstance(value, mx.array):
            tensors[path] = to_torch(value)
            return {"__array__": path, "__dtype__": str(value.dtype).split(".")[-1]}
        if isinstance(value, (bool, int, float, str)) or value is None:
            return {"__scalar__": value}
        if isinstance(value, dict):
            return {"__dict__": [(repr(k), cls._flatten(v, f"{path}{{{k!r}}}", tensors)) for k, v in value.items()],
                    "__keys__": list(value)}
        if isinstance(value, (list, tuple)):
            return {"__seq__": [cls._flatten(v, f"{path}[{i}]", tensors) for i, v in enumerate(value)],
                    "__tuple__": isinstance(value, tuple)}
        if hasattr(value, "__dict__"):
            attrs = {}
            for name, v in vars(value).items():
                flat = cls._flatten(v, f"{path}.{name}", tensors)
                if flat is not None:
                    attrs[name] = flat
            return {"__object__": type(value).__name__, "__attrs__": attrs}
        return None

    @classmethod
    def _restore(cls, template: Any, skeleton: Any, tensors: Mapping[str, torch.Tensor]) -> Any:
        mx = _mx()
        if "__array__" in skeleton:
            name = skeleton["__dtype__"]
            dtype = getattr(mx, name, None) or getattr(mx, name + "_")        # str(mx.bool_) prints "bool"
            return mx.array(tensors[skeleton["__array__"]].numpy(), dtype=dtype)
        if "__scalar__" in skeleton:
            return skeleton["__scalar__"]
        if "__dict__" in skeleton:
            base = template if isinstance(template, dict) else {}
            return {k: cls._restore(base.get(k), v, tensors) for k, (_, v) in zip(skeleton["__keys__"], skeleton["__dict__"])}
        if "__seq__" in skeleton:
            base = list(template) if isinstance(template, (list, tuple)) else []
            items = [cls._restore(base[i] if i < len(base) else None, v, tensors) for i, v in enumerate(skeleton["__seq__"])]
            return tuple(items) if skeleton["__tuple__"] else items
        if "__object__" in skeleton:
            if template is None or type(template).__name__ != skeleton["__object__"]:
                raise ValueError("snapshot carries state the template does not hold; wrong session shape")
            wanted = skeleton["__attrs__"]
            for name in list(vars(template)):
                if name not in wanted and cls._flatten(getattr(template, name), "", {}) is not None:
                    delattr(template, name)
            for name, flat in wanted.items():
                setattr(template, name, cls._restore(getattr(template, name, None), flat, tensors))
            return template
        raise ValueError("malformed snapshot")

    def snapshot_state(self, state: Any) -> dict[str, Any]:
        """Serializable copy of the whole cache list: every array and scalar attribute of every
        per-layer cache object, through nested caches, dicts, lists and tuples."""
        eval_state(state)
        tensors: dict[str, torch.Tensor] = {}
        skeleton = {f"layer{i}": self._flatten(layer, f"layer{i}", tensors) for i, layer in enumerate(state)}
        return {"tensors": tensors, "skeleton": skeleton}

    def restore_state(self, template: Any, snapshot: Mapping[str, Any]) -> Any:
        tensors, skeleton = snapshot["tensors"], snapshot["skeleton"]
        if set(skeleton) != {f"layer{i}" for i in range(len(template))}:
            raise ValueError("snapshot carries state the template does not hold; wrong session shape")
        eval_state(template)
        state = copy.deepcopy(template)
        for i, layer in enumerate(state):
            state[i] = self._restore(layer, skeleton[f"layer{i}"], tensors)
        return state

    def cache_tensors(self, state: Any) -> list[torch.Tensor]:
        """Every cache array as a torch tensor, in a stable order (equality checks)."""
        snapshot = self.snapshot_state(state)
        return [snapshot["tensors"][k] for k in sorted(snapshot["tensors"])]

    def import_self_prefix(self, reference: Any, canonical: Mapping[int, Any], positions: torch.Tensor) -> Any:
        """M0 identity replacement (spec §4.2, D2): KV-bearing layers rebuilt from canonical
        entries at their exact positions; recurrent, conv and indexer state from `reference`."""
        if set(canonical) != set(self.descriptor.kv_layers):
            raise ValueError("self-handoff requires every KV-bearing layer")
        n = positions.numel()
        if n == 0 or not torch.equal(positions.cpu(), torch.arange(n)):
            raise ValueError("self-handoff requires a complete prefix starting at position zero")
        if self.state_length(reference) != n:
            raise ValueError("reference private state length must equal the prefix length")
        eval_state(reference)
        state = copy.deepcopy(reference)
        for layer in self.descriptor.kv_layers:
            k, v = self.native_kv_from_canonical(layer, canonical[layer], positions)
            cache = self._kv_cache(state, layer)
            require_plain_cache(cache)
            keys, values = cache.keys[..., :cache.offset, :], cache.values[..., :cache.offset, :]
            if tuple(k.shape) != tuple(keys.shape) or tuple(v.shape) != tuple(values.shape):
                raise ValueError("rebuilt KV shape mismatch")
            cache.keys, cache.values = k.astype(keys.dtype), v.astype(values.dtype)
        eval_state(state)
        return state


# -- Qwen4Exp attention with capture and the gated foreign block --------------------------

class DriftMlxQwen4ExpAttention:
    """Mixin; bound onto `Qwen4ExpAttention` at substitution time."""

    def _prepare_projected_qkv(self, q_proj_output, keys, values, cache, position_ids, position_embeddings, mask):
        # Canonical boundary: after native key normalization, before rotary; V unrotated.
        # Recomputed on the same projection the stock path normalizes (identical values).
        b, l, _ = keys.shape
        k_can = self.k_norm(keys.reshape(b, l, self.num_key_value_heads, -1))
        v_can = values.reshape(b, l, self.num_key_value_heads, -1)
        self._drift_ctx.capture[self._drift_layer] = (k_can[0], v_can[0])          # [T,Hkv,D]
        return super()._prepare_projected_qkv(q_proj_output, keys, values, cache, position_ids, position_embeddings, mask)

    def __call__(self, x, mask=None, cache=None, position_ids=None, position_embeddings=None):
        mx = _mx()
        ctx, layer = self._drift_ctx, self._drift_layer
        self._drift_last_input = x
        log_prior = ctx.log_prior(layer)
        if log_prior is None:
            return super().__call__(x, mask=mask, cache=cache, position_ids=position_ids,
                                    position_embeddings=position_embeddings)
        if isinstance(mask, mx.array) or (isinstance(mask, str) and mask != "causal"):
            raise ValueError("foreign attention is qualified for unpadded causal batches of one")
        require_plain_cache(cache)
        entries = ctx.foreign.layers[layer]
        if not isinstance(entries, KV):
            raise ValueError("qwen4_exp receives kv_split foreign entries")
        entries.check()
        # Same order as the stock path: indexer state first, then the KV cache update.
        selection = self.indexer.select(x, cache, position_ids)
        b, l, _ = x.shape
        queries, keys, values, gate, _ = self._prepare_projected_qkv(
            self.q_proj(x), self.k_proj(x), self.v_proj(x), cache, position_ids, position_embeddings, None)
        native_tokens = keys.shape[2]
        allowed = _causal_allowed(l, native_tokens) if selection is None else self.indexer.build_mask(selection)
        fk = to_mx(entries.k).transpose(1, 0, 2)[None].astype(keys.dtype)                    # [1,Hkv,Tf,D]
        fv = to_mx(entries.v).transpose(1, 0, 2)[None].astype(values.dtype)
        fk, _ = self.rotary_emb.apply_rotary(fk, fk, ctx.virtual_positions(), unsqueeze_dim=1)   # model's own M-RoPE
        output, foreign_weights = dense_gated_attention(queries, keys, values, allowed, fk, fv, log_prior, self.scale)
        ctx.mass[layer] = foreign_weights[0].sum(-1).T                                         # [L, Hq]
        ctx.entry_mass[layer] = foreign_weights[0].mean(axis=(0, 1))                            # [Tf]
        output = output.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class MlxQwen4ExpAdapter(MlxHybridAdapter):
    def _substitute(self) -> None:
        m = _stock()
        lm = self.model
        cfg = lm.args
        if cfg.model_type != "qwen4_exp_text":
            raise ValueError("expected a qwen4_exp text model")
        if any(t not in {"linear_attention", "qwen_sparse_attention"} for t in cfg.layer_types):
            raise ValueError("unknown layer type; requalify the adapter")
        kv_layers = tuple(i for i, t in enumerate(cfg.layer_types) if t == "qwen_sparse_attention")
        patched = type("DriftMlxQwen4ExpAttentionImpl", (DriftMlxQwen4ExpAttention, m.Qwen4ExpAttention), {})
        for i in kv_layers:
            attention = lm.model.layers[i].self_attn
            if type(attention) is not m.Qwen4ExpAttention:
                raise ValueError("unexpected attention class; requalify the adapter")
            attention.__class__ = patched
            attention._drift_ctx = self.ctx
            attention._drift_layer = i
        self._text = lm.model
        self.descriptor = Descriptor("qwen4_exp", kv_layers, "kv_split", "partial_rope",
                                     cfg.num_key_value_heads, cfg.head_dim)

    def _bind(self) -> None:
        for i in self.descriptor.kv_layers:
            self._text.layers[i].self_attn._drift_ctx = self.ctx

    def _kv_cache(self, state, layer):
        return state[layer]

    def _run(self, inputs, cache, inputs_embeds=None):
        mx = _mx()
        offset = self.state_length(cache)
        length = inputs.shape[1] if inputs_embeds is None else inputs_embeds.shape[1]
        position_ids = mx.arange(offset, offset + length, dtype=mx.int32)[None]              # 2-D text positions
        if inputs is None:
            inputs = mx.zeros((1, length), dtype=mx.int32)      # only read by PLE layers (none qualified)
        hidden = self._text(inputs, inputs_embeds=inputs_embeds, cache=cache, position_ids=position_ids)
        if self.model.args.tie_word_embeddings:
            logits = self._text.embed_tokens.as_linear(hidden)
        else:
            logits = self.model.lm_head(hidden)
        return logits[0]

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        mx = _mx()
        out = self._text.embed_tokens(mx.array(ids.detach().cpu().numpy().astype(np.int32)))
        mx.eval(out)
        return to_torch(out)

    def native_kv_from_canonical(self, layer, entry, positions):
        mx = _mx()
        if not isinstance(entry, KV):
            raise ValueError("qwen4_exp canonical entries are KV")
        entry.check()
        k = to_mx(entry.k).transpose(1, 0, 2)[None]                                          # [1,Hkv,T,D]
        pos = mx.array(positions.cpu().numpy().astype(np.int32))[None]
        rotary = self._text.layers[layer].self_attn.rotary_emb
        k, _ = rotary.apply_rotary(k, k, pos, unsqueeze_dim=1)
        return k, to_mx(entry.v).transpose(1, 0, 2)[None]
