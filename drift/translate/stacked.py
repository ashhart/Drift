"""Stacked-level reader: one member's KV layers predicted from ALL of another member's levels.

Found on real weights (docs/agent-progress.md, 2026-09-19): a per-level linear map between GLM-5.3
latents and Qwen3.8 K/V is capacity-limited (~0.73 relative error however much data it sees), a
map from all writer levels at the token is not (0.57 and falling with data), and the reader's
attention tolerates noise but not shrinkage. Ridge predictions are shrunk by construction, so each
output dimension is rescaled by `gain ** gain_power`, where gain = true std / predicted std measured
on validation rows the regression never saw.

NumPy only so it runs anywhere the taps land (sidecar, Spark host, oMLX runtime).
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class StackedReader:
    writer_layers: tuple[int, ...]            # order in which writer levels are concatenated
    reader_layers: tuple[int, ...]
    kv_heads: int
    head_dim: int
    input_mean: np.ndarray                    # [sum of writer widths]
    weights: Mapping[int, np.ndarray]         # reader layer -> [inputs, 2*kv_heads*head_dim]
    biases: Mapping[int, np.ndarray]
    gains: Mapping[int, np.ndarray]
    meta: dict
    sha256: str

    @classmethod
    def load(cls, path: Path, writer_layers: Sequence[int], reader_layers: Sequence[int], kv_heads: int, head_dim: int) -> "StackedReader":
        z = np.load(path)
        width = 2 * kv_heads * head_dim if kv_heads else head_dim
        weights, biases, gains = ({l: z[f"{name}{l}"].astype(np.float32) for l in reader_layers} for name in ("W", "b", "gain"))
        for l in reader_layers:
            if weights[l].shape != (z["g_mean"].shape[0], width) or biases[l].shape != (width,) or gains[l].shape != (width,):
                raise ValueError(f"layer {l}: artifact shapes do not match the declared reader layout")
            if not (np.isfinite(weights[l]).all() and np.isfinite(gains[l]).all() and (gains[l] > 0).all()):
                raise ValueError(f"layer {l}: nonfinite or nonpositive parameters")
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return cls(tuple(writer_layers), tuple(reader_layers), kv_heads, head_dim, z["g_mean"].astype(np.float32), weights, biases, gains,
                   meta, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def read(self, latents: Mapping[int, np.ndarray], gain_power: float = 1.0) -> dict:
        """latents[writer layer] = [T, width] -> canonical entries per reader layer: (K [T,H,D], V [T,H,D]),
        or a single latent [T, head_dim] when the reader was loaded with kv_heads=0 (MLA)."""
        missing = [l for l in self.writer_layers if l not in latents]
        if missing:
            raise ValueError(f"missing writer layers {missing}")
        lengths = {latents[l].shape[0] for l in self.writer_layers}
        if len(lengths) != 1:
            raise ValueError("writer levels must cover the same tokens")
        x = np.concatenate([np.asarray(latents[l], dtype=np.float32) for l in self.writer_layers], axis=1)
        if x.shape[1] != self.input_mean.shape[0]:
            raise ValueError("writer width does not match the fitted translator")
        if not np.isfinite(x).all():
            raise ValueError("nonfinite writer entries")
        x = x - self.input_mean
        out, half = {}, self.kv_heads * self.head_dim
        for l in self.reader_layers:
            y = (x @ self.weights[l]) * self.gains[l] ** gain_power + self.biases[l]
            out[l] = y if not self.kv_heads else (y[:, :half].reshape(-1, self.kv_heads, self.head_dim), y[:, half:].reshape(-1, self.kv_heads, self.head_dim))
        return out
