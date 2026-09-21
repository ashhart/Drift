"""Adapter-contract wrapper over the kit's FrozenDecoder (toy, qwen3, llama).

Dense models: every layer bears KV in kv_split layout. Private state is the kit's
immutable NativeState, so no deep copy is needed. Exists so the worker, hive
scheduler and mailbox can be tested in the pinned reference environment.
"""
from __future__ import annotations
from types import MappingProxyType
from typing import Any, Mapping
import torch
from drift.core.attention import Gate
from drift.core.memory import ForeignView
from drift.core.types import KV
from drift.runtime.decoder import FrozenDecoder, NativeState
from .base import Descriptor, ForeignEntries, StepOutput


class DenseAdapter:
    def __init__(self, decoder: FrozenDecoder):
        self.decoder = decoder
        self.device = decoder.device
        layers = tuple(range(len(decoder.layers)))
        self.descriptor = Descriptor(decoder.model.config.model_type, layers, "kv_split", "rope",
                                     decoder.kvheads, decoder.dim)

    def state_length(self, state: NativeState | None) -> int:
        return 0 if state is None else state.length

    def forward(self, ids: torch.Tensor, state: NativeState | None = None,
                foreign: ForeignEntries | None = None, gates: Mapping[int, Gate] | None = None,
                override: float | None = None) -> StepOutput:
        view = None
        if foreign is not None:
            unknown = set(foreign.layers) - set(self.descriptor.kv_layers)
            if unknown:
                raise ValueError(f"foreign entries target non-KV layers {sorted(unknown)}")
            for entry in foreign.layers.values():
                if not isinstance(entry, KV):
                    raise ValueError("dense adapters receive kv_split foreign entries")
            view = ForeignView(0, foreign.positions, MappingProxyType(dict(foreign.layers)))
            if override is None and not gates:
                raise ValueError("foreign attention requires a gate or explicit override")
        out = self.decoder.forward(ids, state, foreign=view, gates=gates, override=override)
        return self._wrap(out, view, override)

    def _wrap(self, out, view, override) -> StepOutput:
        live = view is not None and override != 0.0
        mass = {i: m for i, m in out.foreign_mass.items()} if live else {}
        entry = {}
        if live:
            # The kit's attention reports only total mass; spread it uniformly as a
            # diagnostic per-entry estimate for dense toys (hybrid adapters report exact values).
            for i, m in mass.items():
                n = view.positions.numel()
                entry[i] = torch.full((n,), float(m.detach().mean()) / n)
        return StepOutput(out.logits, out.state, dict(out.canonical_delta), mass, entry)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.model.model.embed_tokens(ids.to(self.device))

    def forward_embeddings(self, embeddings: torch.Tensor, state: NativeState | None = None,
                           foreign: ForeignEntries | None = None, gates: Mapping[int, Gate] | None = None,
                           override: float | None = None, ple_ids: torch.Tensor | None = None) -> StepOutput:
        view = None if foreign is None else ForeignView(0, foreign.positions, MappingProxyType(dict(foreign.layers)))
        out = self.decoder.forward(None, state, foreign=view, gates=gates, override=override, embeddings=embeddings)
        return self._wrap(out, view, override)

    def frozen_digest(self) -> str:
        from drift.runtime.manifest import parameter_digest
        return parameter_digest(self.decoder.model)

    def snapshot_state(self, state: NativeState) -> dict[str, torch.Tensor]:
        out = {"length": torch.tensor(state.length)}
        for i, kv in enumerate(state.layers):
            out[f"{i}.k"], out[f"{i}.v"] = kv.k.detach().clone(), kv.v.detach().clone()
        return out

    def restore_state(self, template: Any, snapshot: Mapping[str, torch.Tensor]) -> NativeState:
        length = int(snapshot["length"])
        layers = tuple(KV(snapshot[f"{i}.k"].clone(), snapshot[f"{i}.v"].clone()) for i in range(len(self.decoder.layers)))
        return NativeState(length, layers) if length else NativeState(0, ())

    def import_self_prefix(self, reference: NativeState, canonical: Mapping[int, KV], positions: torch.Tensor) -> NativeState:
        return self.decoder.import_self_prefix(canonical, positions)
