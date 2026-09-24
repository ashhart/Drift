"""A contextual cache reader: GLM's latents over a whole context -> Qwen's full-attention K/V rows, one per GLM token.

Per-token translators map each token on its own, so what a token means in its context (which function a line sits in)
is lost. This reader is a small bidirectional transformer with rotary positions over the sequence of GLM's reduced
latents. It writes a correction onto a fixed per-token rows translator, scaled per output dimension by the targets'
spread; its output layer starts at zero, so an untrained reader is exactly that translator. Both backbones stay frozen.
MLX; the fixed translator and the standardisation are NumPy arrays saved beside the weights.
"""
from __future__ import annotations
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from drift.translate import ridge_map


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.attention_norm, self.mlp_norm = nn.RMSNorm(width), nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.rope = nn.RoPE(width // heads, base=10000)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        tokens, width = x.shape
        q, k, v = mx.split(self.qkv(self.attention_norm(x)), 3, axis=-1)
        q, k, v = (t.reshape(1, tokens, self.heads, width // self.heads).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=(width // self.heads) ** -0.5)   # every token sees the whole context
        x = x + self.out(attended.transpose(0, 2, 1, 3).reshape(tokens, width))
        return x + self.down(nn.gelu(self.up(self.mlp_norm(x))))


class ContextReader(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 1024, layers: int = 4, heads: int = 8):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = [Block(width, heads) for _ in range(layers)]
        self.norm = nn.RMSNorm(width)
        self.head = nn.Linear(width, out_dim)
        self.head.weight = mx.zeros_like(self.head.weight)
        self.head.bias = mx.zeros_like(self.head.bias)

    def __call__(self, z: mx.array) -> mx.array:
        """Standardised reduced latents [tokens, in_dim] -> the correction in units of the targets' spread [tokens, out_dim]."""
        h = self.inp(z)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


class ContextRows:
    """A trained reader with its fixed rows translator: read(features) -> Qwen rows [tokens, out_dim], float32.

    The reader reads the rows translator's reduced features, or with config input "full" GLM's full latents, standardised.
    With gain.npz in the folder (drift/translate/spread.py), the rows are rescaled to the targets' spread. With a window,
    a longer context is read in overlapping windows of that many tokens, as long as those the reader was trained on,
    and each token takes the rows of the window whose centre is nearest it."""

    def __init__(self, folder: Path, window: int | None = None):
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        self.rows = ridge_map.load(folder / "rows.npz")
        arrays = np.load(folder / "scales.npz")
        self.component_scale, self.target_scale = mx.array(arrays["component"]), mx.array(arrays["target"])
        self.input = config.get("input", "reduced")
        if self.input not in ("reduced", "full"):
            raise ValueError(f"unknown reader input {self.input!r}")
        self.input_mean = mx.array(arrays["input_mean"]) if "input_mean" in arrays.files else mx.zeros(arrays["component"].shape)
        self.model = ContextReader(**config["model"])
        self.model.load_weights(str(folder / "reader.safetensors"))
        self.model.eval()
        (weight, bias, gain), = self.rows["layers"].values()
        self.mean, self.basis = mx.array(self.rows["mean"]), mx.array(self.rows["basis"])
        self.weight, self.bias = mx.array(weight * gain[None, :]), mx.array(bias)
        self.spread = None
        if (folder / "gain.npz").exists():
            spread = np.load(folder / "gain.npz")
            if spread["gain"].shape != (config["model"]["out_dim"],) or spread["centre"].shape != spread["gain"].shape:
                raise ValueError("gain.npz does not match the reader's output width")
            self.spread = (mx.array(spread["centre"]), mx.array(spread["gain"]))
        self.window = window or config.get("window")
        if self.window is not None and self.window < 2:
            raise ValueError("a reading window needs at least two tokens")

    def reduce(self, features) -> mx.array:
        return (mx.array(np.asarray(features, np.float32)) - self.mean) @ self.basis

    def _rows(self, x: mx.array) -> mx.array:
        reduced = (x - self.mean) @ self.basis
        seen = x if self.input == "full" else reduced
        return reduced @ self.weight + self.bias + self.model((seen - self.input_mean) / self.component_scale) * self.target_scale

    def read(self, features, spread: bool = True) -> np.ndarray:
        """Rows for every token; spread=False gives the trained output before the gain, for fitting one."""
        x = mx.array(np.asarray(features, np.float32))
        n, w = x.shape[0], self.window
        if w is None or n <= w:
            rows = self._rows(x)
        else:
            starts = np.array(sorted(set(range(0, n - w, w // 2)) | {n - w}))
            owner = np.argmin(np.abs(np.arange(n)[:, None] - (starts + w / 2)[None, :]), axis=1)   # nearest centre; always inside that window
            spans = [(int(s), np.flatnonzero(owner == c)) for c, s in enumerate(starts)]
            rows = mx.concatenate([self._rows(x[s:s + w])[int(own[0]) - s:int(own[-1]) + 1 - s] for s, own in spans if len(own)])   # each window owns one run
        if spread and self.spread is not None:
            centre, gain = self.spread
            rows = centre + gain * (rows - centre)
        mx.eval(rows)
        return np.array(rows, dtype=np.float32)


def row_blocks(width: int) -> list[tuple[int, int]]:
    """Each full-attention layer's K block, then its V block, in the flat row layout of 1,024 values per layer."""
    return [(i * 1024 + off, i * 1024 + off + 512) for i in range(width // 1024) for off in (0, 512)]


def identity_loss(pred: mx.array, target: mx.array, rows: mx.array, scale: mx.array, temperature: float, blocks) -> mx.array:
    """Per K and V block, cross-entropy of each sampled predicted row picking its own token's true row among the sampled
    true rows, by cosine similarity after centring on the true rows' mean and dividing by the targets' spread."""
    total = 0.0
    for lo, hi in blocks:
        t = target[rows][:, lo:hi]
        centre = mx.mean(t, axis=0)
        t, q = (t - centre) / scale[lo:hi], (pred[rows][:, lo:hi] - centre) / scale[lo:hi]
        t = t / mx.maximum(mx.linalg.norm(t, axis=1, keepdims=True), 1e-6)
        q = q / mx.maximum(mx.linalg.norm(q, axis=1, keepdims=True), 1e-6)
        total = total + mx.mean(nn.losses.cross_entropy((q @ t.T) / temperature, mx.arange(t.shape[0])))
    return total / len(blocks)


def identity(pred: mx.array, target: mx.array, blocks) -> mx.array:
    """Per K and V block, the share of predicted rows whose nearest true row in the window, by distance, is their own token's."""
    hits = []
    for lo, hi in blocks:
        q, t = pred[:, lo:hi], target[:, lo:hi]
        distance = mx.sum(q * q, axis=1, keepdims=True) - 2 * (q @ t.T) + mx.sum(t * t, axis=1)[None]
        hits.append(mx.mean((mx.argmin(distance, axis=1) == mx.arange(q.shape[0])).astype(mx.float32)))
    return mx.stack(hits)
