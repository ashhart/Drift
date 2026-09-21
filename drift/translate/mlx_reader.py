"""MLX execution of the corrected fan-out translator and selector-key reader."""
from __future__ import annotations
from typing import Mapping
import numpy as np
from drift.translate.fanout import CorrectedFanoutReader, emission, supported_counts
from drift.translate.index_keys import IndexKeyReader


class MlxForwardReader:
    def __init__(self, reader: CorrectedFanoutReader, index: IndexKeyReader | None, gain_power: float, index_gain_power: float = 1.0):
        import mlx.core as mx
        self.mx, self.reader, b = mx, reader, reader.fan.base
        self.layers, self.heads, self.dim = tuple(b.reader_layers), b.kv_heads, b.head_dim
        half = self.heads * self.dim
        scale = []
        for i, l in enumerate(self.layers):
            g = np.asarray(b.gains[l], dtype=np.float32) ** gain_power
            g = g.copy(); g[:half] *= np.exp(5.0 * reader.loud[i, 0]); g[half:] *= np.exp(5.0 * reader.loud[i, 1])
            scale.append(g)
        self.mean = mx.array(np.asarray(b.input_mean, dtype=np.float32))
        self.W = mx.array(np.concatenate([b.weights[l] for l in self.layers], axis=1).astype(np.float32))
        self.A, self.B = mx.array(reader.A), mx.array(reader.B)
        self.basis = mx.array(reader.fan.basis)
        self.residual = {j: mx.array(r) for j, r in reader.fan.residual.items()}
        bias = [np.asarray(b.biases[l], dtype=np.float32) + (0 if reader.tag is None else reader.tag[i].reshape(-1)) for i, l in enumerate(self.layers)]
        self.scale, self.bias = mx.array(np.concatenate(scale)), mx.array(np.concatenate(bias))
        self.index = index
        if index is not None:
            self.iW, self.ib = mx.array(index.weights * index.gain ** index_gain_power), mx.array(index.bias)
        mx.eval(self.W, self.A, self.B, self.basis, self.scale, self.bias)

    def read(self, latents: Mapping[int, np.ndarray]):
        """-> ({layer: (K, V)} float16 NumPy [rows, heads, dim], {layer: selector keys [rows, index_dim]} or None, order, which); row r is fan-out entry which[r] of writer token order[r]."""
        mx, fan = self.mx, self.reader.fan
        x_np = np.concatenate([np.asarray(latents[l], dtype=np.float32) for l in fan.base.writer_layers], axis=1)
        x = mx.array(x_np) - self.mean
        z = x @ self.basis
        scores = np.array(z @ mx.array(fan.count_w) + mx.array(fan.count_b)); scores[:, 1:] -= fan.margin
        counts = supported_counts(scores, fan.residual)
        order, which = emission(counts)
        take = mx.array(order.astype(np.int32))
        y = (x @ self.W + (x @ self.A) @ self.B)[take]
        for j in np.unique(which[which > 0]):
            rows = mx.array(np.nonzero(which == j)[0].astype(np.int32))
            y[rows] = y[rows] + z[take[rows]] @ self.residual[int(j)]
        y = (y * self.scale + self.bias).astype(mx.float16)
        keys = None
        if self.index is not None:
            keys = ((x @ self.iW)[take] + self.ib).astype(mx.float16)
            mx.eval(y, keys)
        else:
            mx.eval(y)
        y_np, width, half = np.array(y), 2 * self.heads * self.dim, self.heads * self.dim
        out = {l: (y_np[:, i * width:i * width + half].reshape(-1, self.heads, self.dim), y_np[:, i * width + half:(i + 1) * width].reshape(-1, self.heads, self.dim)) for i, l in enumerate(self.layers)}
        if keys is not None:
            k_np, d = np.array(keys), self.index.index_dim
            keys = {l: k_np[:, i * d:(i + 1) * d] for i, l in enumerate(self.index.layers)}
        return out, keys, order, which
