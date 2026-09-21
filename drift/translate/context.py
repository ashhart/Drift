"""Context-augmented forward translation: each reader entry is predicted from the writer's entry at that token PLUS causal
summaries of the writer's earlier entries (exponential moving averages, several horizons, in a PCA subspace).

Why: models bind "which value belongs to which thing" differently. The reader's own key at a value token already carries the
thing it belongs to; the writer may keep that on the EARLIER tokens and reach it by attention. A per-token map cannot move it,
a causal summary can. Streaming-safe: the summary state is carried between chunks. NumPy only."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np


def causal_summaries(z: np.ndarray, decays: Sequence[float], state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """z [T, r] -> ([T, r * len(decays)], final state [len(decays), r]); summary_t = d * summary_{t-1} + (1 - d) * z_t."""
    d = np.asarray(decays, dtype=z.dtype)[:, None]
    s = np.zeros((len(decays), z.shape[1]), z.dtype) if state is None else state.astype(z.dtype).copy()
    out = np.empty((z.shape[0], len(decays), z.shape[1]), z.dtype)
    for t in range(z.shape[0]):
        s = d * s + (1 - d) * z[t]
        out[t] = s
    return out.reshape(z.shape[0], -1), s


@dataclass(frozen=True)
class ContextReader:
    writer_layers: tuple[int, ...]
    reader_layers: tuple[int, ...]
    kv_heads: int
    head_dim: int
    input_mean: np.ndarray            # [D]
    projection: np.ndarray            # [D, r]
    decays: tuple[float, ...]
    summary_scale: float              # summaries are multiplied by this so one ridge penalty suits both feature groups
    feature_mean: np.ndarray          # [features]; the gain acts on the CENTRED prediction, whatever its power
    weights: np.ndarray               # [D + r * len(decays), reader layers * 2 * kv_heads * head_dim]
    bias: np.ndarray
    gain: np.ndarray
    meta: dict
    sha256: str

    @classmethod
    def load(cls, path: Path, writer_layers, reader_layers, kv_heads: int, head_dim: int) -> "ContextReader":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        W, b, g, P, mean, fmean = (z[k].astype(np.float32) for k in ("W", "b", "gain", "P", "g_mean", "f_mean"))
        decays = tuple(float(d) for d in meta["decays"])
        width = 2 * kv_heads * head_dim * len(reader_layers)
        if W.shape != (mean.shape[0] + P.shape[1] * len(decays), width) or P.shape[0] != mean.shape[0] or b.shape != (width,) or g.shape != (width,) or fmean.shape != (W.shape[0],):
            raise ValueError("context translator does not match the declared layers")
        if not all(np.isfinite(a).all() for a in (W, b, g, P, mean, fmean)) or not (g > 0).all() or not all(0 < d < 1 for d in decays):
            raise ValueError("invalid context translator values")
        return cls(tuple(writer_layers), tuple(reader_layers), kv_heads, head_dim, mean, P, decays, float(meta["summary_scale"]), fmean, W, b, g, meta, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def features(self, latents: Mapping[int, np.ndarray], state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([np.asarray(latents[l], dtype=np.float32) for l in self.writer_layers], axis=1) - self.input_mean
        summary, state = causal_summaries(x @ self.projection, self.decays, state)
        return np.concatenate((x, summary * self.summary_scale), axis=1) - self.feature_mean, state

    def read(self, latents: Mapping[int, np.ndarray], gain_power: float = 1.0, state: np.ndarray | None = None) -> dict:
        f, _ = self.features(latents, state)
        y = (f @ self.weights) * self.gain ** gain_power + self.bias
        width, half = 2 * self.kv_heads * self.head_dim, self.kv_heads * self.head_dim
        return {l: (y[:, i * width:i * width + half].reshape(-1, self.kv_heads, self.head_dim), y[:, i * width + half:(i + 1) * width].reshape(-1, self.kv_heads, self.head_dim))
                for i, l in enumerate(self.reader_layers)}


@dataclass(frozen=True)
class ContextFanoutReader:
    """Context base + fan-out (one writer token -> several reader tokens), same arithmetic as FanoutReader on a ContextReader base."""
    base: ContextReader
    basis: np.ndarray
    count_w: np.ndarray
    count_b: np.ndarray
    margin: float
    residual: Mapping[int, np.ndarray]
    sha256: str
    correction: tuple | None = None   # (A [features, r], B [r, outputs], loud [reader layers, 2]) from end-to-end binding training

    def with_correction(self, path: Path) -> "ContextFanoutReader":
        from dataclasses import replace
        from safetensors.numpy import load_file
        w = load_file(str(path))
        A, B, loud = (w[k].astype(np.float32) for k in ("A", "B", "loud"))
        if A.shape[0] != self.base.weights.shape[0] or B.shape != (A.shape[1], self.base.weights.shape[1]) or loud.shape != (len(self.base.reader_layers), 2) or not all(np.isfinite(a).all() for a in (A, B, loud)):
            raise ValueError("correction weights do not match the context base")
        return replace(self, correction=(A, B, loud), sha256=self.sha256 + "+" + hashlib.sha256(Path(path).read_bytes()).hexdigest())

    @classmethod
    def load(cls, path: Path, base: ContextReader) -> "ContextFanoutReader":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        if meta.get("base_sha256") != base.sha256:
            raise ValueError("fan-out maps were fitted on top of a different base translator")
        residual = {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("R")}
        return cls(base, z["basis"].astype(np.float32), z["count_w"].astype(np.float32), z["count_b"].astype(np.float32), float(meta["margin"]), residual, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def read_rows(self, latents: Mapping[int, np.ndarray], gain_power: float = 1.0, state: np.ndarray | None = None):
        """-> ({layer: (K, V)}, order, which, x): reader row r is fan-out entry which[r] of writer token order[r]; x = centred writer rows (for selector keys)."""
        from drift.translate.fanout import emission
        b = self.base
        x = np.concatenate([np.asarray(latents[l], dtype=np.float32) for l in b.writer_layers], axis=1) - b.input_mean
        scores = (x @ self.basis) @ self.count_w + self.count_b
        scores[:, 1:] -= self.margin
        order, which = emission(np.minimum(scores.argmax(1) + 1, max(self.residual, default=0) + 1))
        f, _ = b.features(latents, state)
        y = (f @ b.weights + ((f @ self.correction[0]) @ self.correction[1] if self.correction else 0.0))[order]
        z = None
        for j in np.unique(which[which > 0]):
            z = x @ self.basis if z is None else z
            y[which == j] += z[order[which == j]] @ self.residual[int(j)]
        y = y * b.gain ** gain_power
        width, half = 2 * b.kv_heads * b.head_dim, b.kv_heads * b.head_dim
        if self.correction:
            scale = np.repeat(np.exp(5.0 * self.correction[2]).reshape(-1), half)      # [layer0 K, layer0 V, layer1 K, ...]
            y = y * scale
        y = y + b.bias
        out = {l: (y[:, i * width:i * width + half].reshape(-1, b.kv_heads, b.head_dim), y[:, i * width + half:(i + 1) * width].reshape(-1, b.kv_heads, b.head_dim)) for i, l in enumerate(b.reader_layers)}
        return out, order, which, x
