from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID
import torch
from .types import Delta, KV
from .projector import BridgeProjector


@dataclass(frozen=True)
class ForeignView:
    epoch: int
    positions: torch.Tensor
    layers: Mapping[int, KV]         # RECEIVER layer numbers


class ForeignKVBank:
    """Inference-only, receiver-local, copy-on-publication foreign memory.

    Each delta is projected once. Commit is atomic across selected layers. Readers
    pin the resulting view for an entire forward/decode step and NEVER mutate it.
    This reference is single-threaded; a production worker needs a lock/RCU handoff.
    """
    def __init__(self, session: UUID, direction: int,
                 pairs: list[tuple[int, int, BridgeProjector]],
                 sinks: int = 4, recent: int = 128):
        if not pairs or sinks < 0 or recent <= 0 or direction not in (0, 1):
            raise ValueError("invalid bank configuration")
        if len({t for _, t, _ in pairs}) != len(pairs):
            raise ValueError("each receiver layer needs exactly one selected source")
        self.session, self.direction = session, direction
        self.pairs, self.sinks, self.recent = tuple(pairs), sinks, recent
        self.next_sequence, self.next_start, self.last_epoch = 0, 0, -1
        self._view: ForeignView | None = None

    @torch.no_grad()
    def commit(self, delta: Delta) -> None:
        delta.check()
        if delta.session != self.session or delta.direction != self.direction:
            raise ValueError("wrong run/session or direction")
        if delta.sequence != self.next_sequence or delta.start != self.next_start:
            raise ValueError("duplicate, gap, reorder, or missing full-reset handshake")
        if delta.epoch <= self.last_epoch:
            raise ValueError("nonmonotonic publication epoch")
        expected = {s for s, _, _ in self.pairs}
        if set(delta.layers) != expected:
            raise ValueError("publication is missing or adding a bridged layer")
        converted: dict[int, KV] = {}
        for source, target, projector in self.pairs:
            kv = delta.layers[source]
            parameter = next(projector.parameters())
            kv = KV(kv.k.to(parameter), kv.v.to(parameter))
            projected = projector(kv)
            projected.check()
            converted[target] = projected.clone()
        device = next(iter(converted.values())).k.device
        positions = torch.arange(delta.start, delta.start + delta.tokens, device=device)
        if self._view is not None:
            positions = torch.cat((self._view.positions, positions))
            converted = {t: KV(torch.cat((self._view.layers[t].k, kv.k)),
                               torch.cat((self._view.layers[t].v, kv.v)))
                         for t, kv in converted.items()}
        # First source positions are sinks; keep order and preserve real distances.
        keep = (positions < self.sinks) | (positions >= positions[-1] - self.recent + 1)
        trimmed = {t: KV(kv.k[keep], kv.v[keep]) for t, kv in converted.items()}
        new_view = ForeignView(delta.epoch, positions[keep], MappingProxyType(trimmed))
        # All validation/projection/allocation happened before these state updates.
        self._view = new_view
        self.next_sequence += 1
        self.next_start += delta.tokens
        self.last_epoch = delta.epoch

    def pin(self, *, now_epoch: int | None = None,
            max_age: int | None = None) -> ForeignView | None:
        if max_age is not None:
            if now_epoch is None or max_age < 0:
                raise ValueError("TTL requires a current epoch and nonnegative age")
            if self._view is not None and now_epoch - self._view.epoch > max_age:
                return None
        return self._view
