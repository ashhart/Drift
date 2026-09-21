from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID
import torch


@dataclass(frozen=True)
class KV:
    """Canonical K is pre-RoPE, AFTER native K normalization; V is unrotated.

    Layout is [sequence, kv_heads, head_dim]. Batch size is deliberately one.
    Frozen dataclasses do not make tensors immutable: ownership boundaries clone.
    """
    k: torch.Tensor
    v: torch.Tensor

    def check(self) -> None:
        if self.k.ndim != 3 or self.k.shape != self.v.shape:
            raise ValueError("K/V must share [sequence, kv_heads, head_dim]")
        if min(self.k.shape) <= 0:
            raise ValueError("empty K/V blocks are not allowed")
        if self.k.dtype != self.v.dtype or self.k.device != self.v.device:
            raise ValueError("K/V dtype and device must match")
        if not self.k.is_floating_point() or not self.v.is_floating_point():
            raise TypeError("K/V must be floating-point activations")
        if not (torch.isfinite(self.k).all() and torch.isfinite(self.v).all()):
            raise ValueError("nonfinite activation")

    def clone(self, *, detach: bool = True) -> KV:
        f = (lambda x: x.detach().clone()) if detach else (lambda x: x.clone())
        return KV(f(self.k), f(self.v))

    @property
    def tokens(self) -> int:
        return self.k.shape[0]


@dataclass(frozen=True)
class Delta:
    """One complete, contiguous source block for ALL agreed bridged layers.

    Session UUID, direction and layer shapes bind to a separately pinned manifest.
    No text, token IDs, answer labels or free-form metadata are represented here.
    """
    session: UUID
    direction: int                 # 0: A -> B; 1: B -> A
    epoch: int
    sequence: int
    start: int                     # source slot / native source position
    layers: Mapping[int, KV]

    def check(self) -> None:
        if self.direction not in (0, 1):
            raise ValueError("bad direction")
        if min(self.epoch, self.sequence, self.start) < 0 or not self.layers:
            raise ValueError("negative counter or empty layer set")
        if any(type(i) is not int or not 0 <= i <= 65535 for i in self.layers):
            raise ValueError("bad layer index")
        for kv in self.layers.values():
            kv.check()
        if len({x.tokens for x in self.layers.values()}) != 1:
            raise ValueError("a publication must be complete across layers")

    @property
    def tokens(self) -> int:
        return next(iter(self.layers.values())).tokens

    def clone(self) -> Delta:
        return Delta(self.session, self.direction, self.epoch, self.sequence,
                     self.start, {i: x.clone() for i, x in self.layers.items()})
