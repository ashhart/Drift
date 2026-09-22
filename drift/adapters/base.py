"""Model-agnostic adapter contract (docs/guides/ADAPTERS.md) for hybrid HF decoders.

An adapter owns one frozen model instance. It substitutes the attention modules of
the KV-bearing layers with subclasses that run the stock math unchanged and add
two things: canonical capture of the new entries, and a gated foreign block inside
the same softmax. Linear-attention layers, MoE, hyper-connections and sparse
indexers stay stock. Nothing writes into the native cache from outside.

The private native state is the stock cache object (recurrent, conv, indexer and
KV state together). `forward` deep-copies it so an input state stays valid for
counterfactual runs; a production worker advances one owned state instead.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping
import torch
from torch import nn
from torch.nn import functional as F
from drift.core.attention import Gate
from drift.core.position import recency_positions
from drift.core.types import KV


@dataclass(frozen=True)
class Descriptor:
    model_type: str
    kv_layers: tuple[int, ...]
    layout: str                 # "kv_split" ([T,H,D] K and V) or "mla_latent" ([T,C])
    positional: str             # "partial_rope", "rope" or "none"
    kv_heads: int
    head_dim: int               # latent width for mla_latent


@dataclass(frozen=True)
class ForeignEntries:
    """Foreign memory already projected into THIS receiver's canonical layout.

    `positions` are source slots (strictly increasing). `layers` is keyed by the
    receiver's layer index and holds a KV for kv_split or a [T,C] tensor for mla_latent.
    """
    positions: torch.Tensor
    layers: Mapping[int, KV | torch.Tensor]


@dataclass(frozen=True)
class StepOutput:
    output: torch.Tensor                 # [T, vocab] logits, or [T, hidden] when the head is absent
    state: Any                           # private native state (stock cache)
    canonical: Mapping[int, KV | torch.Tensor]
    foreign_mass: Mapping[int, torch.Tensor]          # [Q, Hq] per layer
    entry_mass: Mapping[int, torch.Tensor] = field(default_factory=dict)   # [Tf] per layer, mean over Q and heads


@dataclass
class _Context:
    """Per-forward instructions shared with the substituted attention modules."""
    foreign: ForeignEntries | None = None
    gates: Mapping[int, Gate] | None = None
    override: float | None = None
    first_query_position: int = 0
    capture: dict[int, KV | torch.Tensor] = field(default_factory=dict)
    mass: dict[int, torch.Tensor] = field(default_factory=dict)
    entry_mass: dict[int, torch.Tensor] = field(default_factory=dict)

    def log_prior(self, layer: int) -> float | torch.Tensor | None:
        """None means this layer attends natively only."""
        if self.foreign is None or layer not in self.foreign.layers or self.override == 0.0:
            return None
        if self.override is not None:
            if not 0.0 < self.override <= 1.0:
                raise ValueError("gate override must be in (0,1]; 0 is the exact native path")
            return math.log(self.override)
        gate = None if self.gates is None else self.gates.get(layer)
        if gate is None:
            raise ValueError("foreign attention requires a gate or explicit override")
        return F.logsigmoid(gate.logit)

    def virtual_positions(self, device: torch.device) -> torch.Tensor:
        assert self.foreign is not None
        return recency_positions(self.foreign.positions.to(device), self.first_query_position)


def append_foreign_columns(mask: torch.Tensor, foreign_tokens: int, log_prior) -> torch.Tensor:
    """Extend an additive [B,1,Q,KV] mask with always-visible foreign columns carrying log g."""
    if not mask.is_floating_point():
        raise ValueError("adapters run the eager path with an additive float mask")
    prior = torch.as_tensor(log_prior, dtype=mask.dtype, device=mask.device)
    extra = prior.expand(*mask.shape[:-1], foreign_tokens)
    return torch.cat((mask, extra), dim=-1)


class HybridAdapter:
    """Shared machinery; subclasses supply `_substitute()` and `_run()`."""
    descriptor: Descriptor

    def __init__(self, model: nn.Module):
        model.eval().requires_grad_(False)
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError("set config._attn_implementation = 'eager' before adapting")
        self.model = model
        self.ctx = _Context()
        self.device = next(model.parameters()).device
        self._substitute()

    # -- subclass hooks ------------------------------------------------------
    def _substitute(self) -> None:
        raise NotImplementedError

    def _run(self, ids: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        """Run the stock model on [1,T] ids with a cache; return ([T,X] output, cache)."""
        raise NotImplementedError

    def _run_embeds(self, embeddings: torch.Tensor, state: Any, ple_ids: torch.Tensor | None) -> tuple[torch.Tensor, Any]:
        """Run the stock model on [1,T,hidden] embeddings (mailbox marker path)."""
        raise NotImplementedError

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_length(self, state: Any) -> int:
        return 0 if state is None else int(state.get_seq_length())

    # -- contract ------------------------------------------------------------
    def forward(self, ids: torch.Tensor, state: Any = None, foreign: ForeignEntries | None = None,
                gates: Mapping[int, Gate] | None = None, override: float | None = None) -> StepOutput:
        if ids.ndim != 1 or ids.numel() == 0 or ids.dtype != torch.long:
            raise ValueError("local token IDs must be nonempty int64 [T]")
        if foreign is not None:
            unknown = set(foreign.layers) - set(self.descriptor.kv_layers)
            if unknown:
                raise ValueError(f"foreign entries target non-KV layers {sorted(unknown)}")
        state = None if state is None else copy.deepcopy(state)
        self.ctx = _Context(foreign, gates, override, self.state_length(state))
        try:
            output, state = self._run(ids.to(self.device), state)
        finally:
            ctx, self.ctx = self.ctx, _Context()
        return StepOutput(output, state, dict(ctx.capture), dict(ctx.mass), dict(ctx.entry_mass))

    def forward_embeddings(self, embeddings: torch.Tensor, state: Any = None,
                           foreign: ForeignEntries | None = None, gates: Mapping[int, Gate] | None = None,
                           override: float | None = None, ple_ids: torch.Tensor | None = None) -> StepOutput:
        """Same as `forward` but from [T,hidden] embeddings (learned mailbox marker path)."""
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("local embeddings must be nonempty [T,hidden]")
        state = None if state is None else copy.deepcopy(state)
        if state is not None:
            self._assign_instead_of_copy(state)
        self.ctx = _Context(foreign, gates, override, self.state_length(state))
        try:
            output, state = self._run_embeds(embeddings.to(self.device), state, ple_ids)
        finally:
            ctx, self.ctx = self.ctx, _Context()
        return StepOutput(output, state, dict(ctx.capture), dict(ctx.mass), dict(ctx.entry_mass))

    @staticmethod
    def _assign_instead_of_copy(state: Any) -> None:
        """Private-branch caches are discarded, so replace the stock in-place `copy_`
        recurrent/conv updates (which break autograd into the marker) with assignment.
        Instance-level patches on a deep copy; the owned native state is never touched."""
        import types
        for layer in state.layers:
            if not hasattr(layer, "recurrent_states"):
                continue

            def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
                self.recurrent_states[state_idx] = recurrent_states
                self.is_recurrent_states_initialized[state_idx] = True
                self.has_previous_state[state_idx] = True
                return recurrent_states

            layer.update_recurrent_state = types.MethodType(update_recurrent_state, layer)
            original = layer.update_conv_state

            def update_conv_state(self, conv_states, state_idx=0, **kwargs):
                # Keep the stock bookkeeping, then replace the stored tensor non-destructively.
                stored = self.conv_states.get(state_idx) if isinstance(self.conv_states, dict) else None
                if stored is not None:
                    self.conv_states[state_idx] = stored.clone()
                return original(conv_states, state_idx=state_idx, **kwargs)

            layer.update_conv_state = types.MethodType(update_conv_state, layer)

    def frozen_digest(self) -> str:
        from drift.runtime.manifest import parameter_digest
        return parameter_digest(self.model)

    # -- private state --------------------------------------------------------
    @staticmethod
    def _holders(state: Any):
        """The cache object and each of its layers, keyed for the snapshot."""
        yield "cache", state
        for index, layer in enumerate(state.layers):
            yield f"layer{index}", layer

    @staticmethod
    def _flatten(value: Any, path: str, tensors: dict[str, torch.Tensor]) -> Any:
        """Skeleton mirroring `value` with tensors replaced by references; None for unsupported."""
        if isinstance(value, torch.Tensor):
            tensors[path] = value.detach().clone()
            return {"__tensor__": path}
        if isinstance(value, (bool, int, float, str)) or value is None:
            return {"__scalar__": value}
        if isinstance(value, dict):
            return {"__dict__": [(repr(k), HybridAdapter._flatten(v, f"{path}{{{k!r}}}", tensors)) for k, v in value.items()],
                    "__keys__": [k for k in value]}
        if isinstance(value, (list, tuple)):
            return {"__seq__": [HybridAdapter._flatten(v, f"{path}[{i}]", tensors) for i, v in enumerate(value)],
                    "__tuple__": isinstance(value, tuple)}
        return None                       # device/dtype and other objects stay as in the template

    @staticmethod
    def _unflatten(skeleton: Any, tensors: Mapping[str, torch.Tensor]) -> Any:
        if "__tensor__" in skeleton:
            return tensors[skeleton["__tensor__"]].clone()
        if "__scalar__" in skeleton:
            return skeleton["__scalar__"]
        if "__dict__" in skeleton:
            return {k: HybridAdapter._unflatten(v, tensors) for k, (_, v) in zip(skeleton["__keys__"], skeleton["__dict__"])}
        items = [HybridAdapter._unflatten(v, tensors) for v in skeleton["__seq__"]]
        return tuple(items) if skeleton["__tuple__"] else items

    def snapshot_state(self, state: Any) -> dict[str, Any]:
        """Serializable copy of the whole stock cache: every tensor/scalar attribute on the
        cache object and its layers, through dicts, lists and tuples (M2.3 checkpoints)."""
        tensors: dict[str, torch.Tensor] = {}
        skeleton: dict[str, dict[str, Any]] = {}
        for prefix, holder in self._holders(state):
            skeleton[prefix] = {}
            for name, value in vars(holder).items():
                if name == "layers":
                    continue
                flat = self._flatten(value, f"{prefix}.{name}", tensors)
                if flat is not None:
                    skeleton[prefix][name] = flat
        return {"tensors": tensors, "skeleton": skeleton}

    def restore_state(self, template: Any, snapshot: Mapping[str, Any]) -> Any:
        """Rebuild a cache from `snapshot` using `template` (a cache of the same session shape).
        Snapshotted attributes the template lacks are added; template attributes of a
        snapshotted kind that the snapshot lacks are removed."""
        tensors, skeleton = snapshot["tensors"], snapshot["skeleton"]
        state = copy.deepcopy(template)
        holders = dict(self._holders(state))
        if set(skeleton) != set(holders):
            raise ValueError("snapshot carries state the template does not hold; wrong session shape")
        for prefix, holder in holders.items():
            wanted = skeleton[prefix]
            for name in list(vars(holder)):
                if name == "layers" or name in wanted:
                    continue
                if self._flatten(getattr(holder, name), "", {}) is not None:
                    delattr(holder, name)
            for name, flat in wanted.items():
                setattr(holder, name, self._unflatten(flat, tensors))
        return state

    def native_kv_from_canonical(self, layer: int, entry: KV | torch.Tensor,
                                 positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stock-cache K/V ([1,H,T,*]) for `layer` from canonical entries at exact positions."""
        raise NotImplementedError

    def import_self_prefix(self, reference: Any, canonical: Mapping[int, KV | torch.Tensor],
                           positions: torch.Tensor) -> Any:
        """M0 identity replacement for a hybrid model (spec §4.2, D2).

        The KV-bearing layers are rebuilt from canonical entries at their exact
        positions; the recurrent, conv and indexer state (which has no per-token
        form) is taken from `reference`. Correctness gate only, never a
        cross-family memory path.
        """
        if set(canonical) != set(self.descriptor.kv_layers):
            raise ValueError("self-handoff requires every KV-bearing layer")
        n = positions.numel()
        if n == 0 or not torch.equal(positions.cpu(), torch.arange(n)):
            raise ValueError("self-handoff requires a complete prefix starting at position zero")
        if self.state_length(reference) != n:
            raise ValueError("reference private state length must equal the prefix length")
        state = copy.deepcopy(reference)
        for layer in self.descriptor.kv_layers:
            k, v = self.native_kv_from_canonical(layer, canonical[layer], positions.to(self.device))
            cache_layer = state.layers[layer]
            if k.shape != cache_layer.keys.shape or v.shape != cache_layer.values.shape:
                raise ValueError("rebuilt KV shape mismatch")
            cache_layer.keys, cache_layer.values = k, v
        return state
