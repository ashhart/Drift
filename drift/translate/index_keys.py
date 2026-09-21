"""Selector-key reader: the writer's stacked entries -> the reader's raw sparse-selector keys, one per emitted reader row.

Both Drift models keep only their top-scoring entries once a context exceeds the selector budget, and they choose them with a
separate key per token. An injected entry whose selector key is zero is never chosen (measured: 0% vs 75-83% at 4k-16k tokens),
so long memories need these keys translated as well. Fan-out rows reuse their writer token's prediction."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class IndexKeyReader:
    weights: np.ndarray               # [inputs, layers * index_dim]
    bias: np.ndarray
    gain: np.ndarray
    index_dim: int
    layers: tuple[int, ...]
    meta: dict
    sha256: str

    @classmethod
    def load(cls, path: Path, layers, base_sha256: str) -> "IndexKeyReader":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        if meta.get("base_sha256") != base_sha256:
            raise ValueError("selector-key translator was fitted against a different base translator (input centring differs)")
        W, b, g = (z[k].astype(np.float32) for k in ("W", "b", "gain"))
        dim = int(meta["index_dim"])
        if W.shape[1] != dim * len(layers) or b.shape != (W.shape[1],) or g.shape != b.shape or not all(np.isfinite(a).all() for a in (W, b, g)):
            raise ValueError("selector-key artifact does not match the declared reader layers")
        return cls(W, b, g, dim, tuple(layers), meta, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def read(self, centred_rows: np.ndarray, order: np.ndarray | None = None, gain_power: float = 1.0) -> dict[int, np.ndarray]:
        """centred_rows: the writer's stacked entries minus the base translator's input mean [T, inputs]; `order` maps each
        emitted reader row to its writer row (fan-out). Returns {reader layer: [rows, index_dim]}."""
        x = np.asarray(centred_rows, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[0] or not np.isfinite(x).all():
            raise ValueError("writer rows do not match the selector-key translator")
        y = (x @ self.weights) * self.gain ** gain_power + self.bias
        if order is not None:
            y = y[np.asarray(order)]
        return {layer: y[:, i * self.index_dim:(i + 1) * self.index_dim] for i, layer in enumerate(self.layers)}
