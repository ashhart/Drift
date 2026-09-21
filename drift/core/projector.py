from __future__ import annotations
import torch
from torch import nn
from .types import KV


class LinearMap(nn.Module):
    def __init__(self, source: tuple[int, int], target: tuple[int, int]):
        super().__init__()
        self.source, self.target = source, target
        self.map = nn.Linear(source[0] * source[1], target[0] * target[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.source:
            raise ValueError("projector source shape mismatch")
        return self.map(x.flatten(1)).reshape(x.shape[0], *self.target)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> None:
        """Offline FP64 centered ridge; separate intercept, not regularized.

        This simple primal solver is appropriate for a bounded reference dataset.
        Stream X'X/X'Y or use a stable factorization for large-scale fitting.
        """
        if ridge <= 0 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise ValueError("need paired rows and positive ridge")
        if tuple(x.shape[1:]) != self.source or tuple(y.shape[1:]) != self.target:
            raise ValueError("ridge shape mismatch")
        a, b = x.detach().cpu().double().flatten(1), y.detach().cpu().double().flatten(1)
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("nonfinite training data")
        am, bm = a.mean(0), b.mean(0)
        ac, bc = a - am, b - bm
        lhs = ac.T @ ac + ridge * torch.eye(ac.shape[1], dtype=torch.float64)
        weight = torch.linalg.solve(lhs, ac.T @ bc)
        self.map.weight.copy_(weight.T.to(self.map.weight))
        self.map.bias.copy_((bm - am @ weight).to(self.map.bias))


class HeadMap(nn.Module):
    """Learned head mixing + channel map + normalized residual MLP.

    The residual preserves a direct amplitude-carrying path; K and V do not share
    weights. A baseline map is not evidence that this architecture will transfer.
    """
    def __init__(self, source: tuple[int, int], target: tuple[int, int], rank: int = 64):
        super().__init__()
        self.source, self.target = source, target
        hs, ds = source
        ht, dt = target
        self.mix = nn.Parameter(torch.randn(ht, hs) / hs**0.5)
        self.channel = nn.Linear(ds, dt)
        self.residual = nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, rank),
                                      nn.SiLU(), nn.Linear(rank, dt))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.source:
            raise ValueError("head-map shape mismatch")
        mixed = torch.einsum("oh,thd->tod", self.mix, x)
        return self.channel(mixed) + self.residual(mixed)


class BridgeProjector(nn.Module):
    def __init__(self, source: tuple[int, int], target: tuple[int, int],
                 kind: str = "ridge", rank: int = 64):
        super().__init__()
        if kind not in {"ridge", "mlp"}:
            raise ValueError("kind must be ridge or mlp")
        self.source, self.target, self.kind, self.rank = source, target, kind, rank
        factory = ((lambda: LinearMap(source, target)) if kind == "ridge"
                   else (lambda: HeadMap(source, target, rank)))
        self.k_map, self.v_map = factory(), factory()

    def forward(self, kv: KV) -> KV:
        kv.check()
        return KV(self.k_map(kv.k), self.v_map(kv.v))

    def fit(self, source: KV, target: KV, ridge: float = 1e-3) -> None:
        if self.kind != "ridge":
            raise ValueError("use the optimizer training stage for MLP maps")
        self.k_map.fit(source.k, target.k, ridge)
        self.v_map.fit(source.v, target.v, ridge)
