"""Model pack: one member's writer into pool.v2 and reader out of it (docs/research/HIVE_MIND.md, H1/H9).

pool.v2 holds ONE vector per token. A writer maps the member's stacked canonical entries (all of its KV-bearing
levels at that token) into the pool; a reader maps pool rows written by any other member to each of its own
layers, rescaling every output dimension by `gain ** gain_power` because regression shrinks and attention keys
cannot tolerate shrinkage (docs/agent-progress.md, 2026-09-19). Packs only interoperate when their pool
fingerprints match; a pack is tied to the checkpoint/quantization/cache dtype it was fitted on (meta).

NumPy only, so a pack can be applied wherever the taps land (sidecar, serving host, oMLX runtime)."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ModelPack:
    member: str
    layout: str                     # "kv_split" (K and V [T,H,D] per layer) or "mla_latent" ([T,width] per layer)
    layers: tuple[int, ...]
    layer_width: int                # flattened width of one layer's entry
    kv_heads: int
    pool_width: int
    pool_fingerprint: str
    writer: np.ndarray              # [layers*layer_width, pool_width]
    reader: np.ndarray              # [pool_width, layers*layer_width]
    gain: np.ndarray
    own_mean: np.ndarray
    meta: dict
    sha256: str

    @classmethod
    def load(cls, path: Path, kv_heads: int = 0) -> "ModelPack":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        layers, width, layout = tuple(int(l) for l in meta["layers"]), int(meta["width"]), meta["layout"]
        own, pool = len(layers) * width, int(meta["pool"]["width"])
        writer, reader, gain, mean = (z[k].astype(np.float32) for k in ("writer", "reader", "gain", "own_mean"))
        if writer.shape != (own, pool) or reader.shape != (pool, own) or gain.shape != (own,) or mean.shape != (own,):
            raise ValueError("pack arrays do not match the layout declared in its metadata")
        if layout not in {"kv_split", "mla_latent"} or (layout == "kv_split" and (kv_heads <= 0 or width % (2 * kv_heads))):
            raise ValueError("kv_split packs need the reader's KV head count; mla_latent packs need none")
        if not all(np.isfinite(a).all() for a in (writer, reader, gain, mean)) or (gain <= 0).any():
            raise ValueError("nonfinite or nonpositive pack parameters")
        return cls(meta["member"], layout, layers, width, kv_heads, pool, meta["pool"]["fingerprint"], writer, reader, gain, mean, meta,
                   hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def _stack(self, entries: Mapping[int, object]) -> np.ndarray:
        missing = [l for l in self.layers if l not in entries]
        if missing:
            raise ValueError(f"missing layers {missing}; a partial token is never written")
        parts = []
        for l in self.layers:
            e = entries[l]
            flat = np.concatenate((np.asarray(e[0]).reshape(len(e[0]), -1), np.asarray(e[1]).reshape(len(e[1]), -1)), axis=1) if self.layout == "kv_split" else np.asarray(e)
            if flat.ndim != 2 or flat.shape[1] != self.layer_width:
                raise ValueError(f"layer {l}: entry width {flat.shape} does not match the pack ({self.layer_width})")
            parts.append(flat.astype(np.float32))
        if len({p.shape[0] for p in parts}) != 1:
            raise ValueError("layers must cover the same tokens")
        x = np.concatenate(parts, axis=1)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite entries")
        return x

    def write(self, entries: Mapping[int, object]) -> np.ndarray:
        """Canonical entries of this member -> pool rows [T, pool_width]."""
        return (self._stack(entries) - self.own_mean) @ self.writer

    def read(self, pool_rows: np.ndarray, writer_fingerprint: str, gain_power: float = 1.0) -> dict:
        """Pool rows written by another member -> this member's canonical entries per layer."""
        if writer_fingerprint != self.pool_fingerprint:
            raise ValueError("pool fingerprints differ: these packs were fitted against different pool formats; re-enroll")
        rows = np.asarray(pool_rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.pool_width or not np.isfinite(rows).all():
            raise ValueError("pool rows must be finite [T, pool_width]")
        y = (rows @ self.reader) * self.gain ** gain_power + self.own_mean
        out = {}
        for i, l in enumerate(self.layers):
            part = y[:, i * self.layer_width:(i + 1) * self.layer_width]
            if self.layout == "kv_split":
                half = self.layer_width // 2
                out[l] = (part[:, :half].reshape(len(part), self.kv_heads, -1), part[:, half:].reshape(len(part), self.kv_heads, -1))
            else:
                out[l] = part
        return out
