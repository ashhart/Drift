"""Translate tokenization differences using a span classifier and frozen residual maps."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
from drift.translate.stacked import StackedReader


def emission(counts) -> tuple[np.ndarray, np.ndarray]:
    """Reader-row order for predicted spans: for writer row t spanning n reader tokens, rows (t, n-1) ... (t, 0)."""
    order, which = [], []
    for t, n in enumerate(counts):
        for j in range(int(n) - 1, -1, -1):
            order.append(t); which.append(j)
    return np.asarray(order), np.asarray(which)


def supported_counts(scores, residual):
    """Clamp emissions to the contiguous residual prefix shared by both runtimes."""
    count = 1
    while count in residual:
        count += 1
    return np.minimum(np.asarray(scores).argmax(1) + 1, count)


@dataclass(frozen=True)
class FanoutReader:
    base: StackedReader
    basis: np.ndarray                 # [inputs, r] shared input subspace for the classifier and the residual maps
    count_w: np.ndarray               # [r, classes]; class c means the writer token spans c+1 reader tokens
    count_b: np.ndarray
    margin: float                     # a span > 1 is predicted only when its score beats "1" by this margin
    residual: Mapping[int, np.ndarray]   # j -> [r, outputs] over ALL reader layers, in base.reader_layers order
    meta: dict
    sha256: str

    @classmethod
    def load(cls, path: Path, base: StackedReader) -> "FanoutReader":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        if meta.get("base_sha256") != base.sha256:
            raise ValueError("fan-out maps were fitted on top of a different base translator")
        residual = {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("R")}
        return cls(base, z["basis"].astype(np.float32), z["count_w"].astype(np.float32), z["count_b"].astype(np.float32), float(meta["margin"]), residual, meta,
                   hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def spans(self, x: np.ndarray) -> np.ndarray:
        """x: centred stacked writer rows [T, inputs] -> predicted reader-token count per writer token (>= 1)."""
        scores = (x @ self.basis) @ self.count_w + self.count_b
        scores[:, 1:] -= self.margin
        return supported_counts(scores, self.residual)

    def read(self, latents: Mapping[int, np.ndarray], gain_power: float = 1.0) -> dict:
        b = self.base
        x = np.concatenate([np.asarray(latents[l], dtype=np.float32) for l in b.writer_layers], axis=1) - b.input_mean
        z = None
        order, which = emission(self.spans(x))                       # reader row -> (writer row, j)
        width, out = 2 * b.kv_heads * b.head_dim, {}
        for i, l in enumerate(b.reader_layers):
            y = x[order] @ b.weights[l]
            for j in np.unique(which[which > 0]):
                rows = which == j
                z = x @ self.basis if z is None else z
                y[rows] += z[order[rows]] @ self.residual[int(j)][:, i * width:(i + 1) * width]
            y = y * b.gains[l] ** gain_power + b.biases[l]
            half = width // 2
            out[l] = (y[:, :half].reshape(-1, b.kv_heads, b.head_dim), y[:, half:].reshape(-1, b.kv_heads, b.head_dim))
        return out


@dataclass(frozen=True)
class CorrectedFanoutReader:
    """Apply frozen fan-out residuals, trained corrections and optional source tags."""
    fan: FanoutReader
    A: np.ndarray                     # [inputs, rank]
    B: np.ndarray                     # [rank, outputs over all reader layers]
    loud: np.ndarray                  # [reader layers, 2]
    sha256: str
    tag: np.ndarray | None = None     # [reader layers, 2, kv_heads * head_dim]

    @classmethod
    def load(cls, path: Path, fan: FanoutReader) -> "CorrectedFanoutReader":
        from safetensors.numpy import load_file
        w = load_file(str(path))
        A, B, loud = (w[k].astype(np.float32) for k in ("A", "B", "loud"))
        width = 2 * fan.base.kv_heads * fan.base.head_dim
        if A.shape[0] != fan.base.input_mean.shape[0] or B.shape != (A.shape[1], width * len(fan.base.reader_layers)) or loud.shape != (len(fan.base.reader_layers), 2):
            raise ValueError("correction weights do not match the base translator's layout")
        tag = w["tag"].astype(np.float32) if "tag" in w else None
        if tag is not None and tag.shape != (len(fan.base.reader_layers), 2, width // 2):
            raise ValueError("source tag does not match the base translator's layout")
        if not all(np.isfinite(x).all() for x in (A, B, loud) + (() if tag is None else (tag,))):
            raise ValueError("nonfinite correction weights")
        return cls(fan, A, B, loud, hashlib.sha256(Path(path).read_bytes()).hexdigest(), tag)

    def read(self, latents: Mapping[int, np.ndarray], gain_power: float = 1.0) -> dict:
        b = self.fan.base
        x = np.concatenate([np.asarray(latents[l], dtype=np.float32) for l in b.writer_layers], axis=1) - b.input_mean
        order, which = emission(self.fan.spans(x))
        z, corr = x @ self.fan.basis, (x @ self.A) @ self.B
        width, out = 2 * b.kv_heads * b.head_dim, {}
        for i, l in enumerate(b.reader_layers):
            y = x[order] @ b.weights[l] + corr[order][:, i * width:(i + 1) * width]
            for j in np.unique(which[which > 0]):
                rows = which == j
                y[rows] += z[order[rows]] @ self.fan.residual[int(j)][:, i * width:(i + 1) * width]
            y = y * b.gains[l] ** gain_power
            half = width // 2
            y[:, :half] *= np.exp(5.0 * self.loud[i, 0]); y[:, half:] *= np.exp(5.0 * self.loud[i, 1])
            if self.tag is not None:
                y[:, :half] += self.tag[i, 0]; y[:, half:] += self.tag[i, 1]
            y = y + b.biases[l]
            out[l] = (y[:, :half].reshape(-1, b.kv_heads, b.head_dim), y[:, half:].reshape(-1, b.kv_heads, b.head_dim))
        return out
