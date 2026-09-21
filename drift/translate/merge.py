"""Tokenisation-aware reverse translation (several writer tokens -> one reader token).

When the writer splits text more finely than the reader (Qwen3.8 `4|2|7` vs GLM-5.3 `4|27`), translating every writer
token hands the reader stray entries for positions that are not tokens in its own vocabulary. MergeFilter drops them,
text-free: a linear classifier on the writer's stacked entry and the NEXT token's entry (whether "2" ends a reader
token depends on what follows) predicts which writer tokens end a reader token; only those are translated."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class MergeFilter:
    mean: np.ndarray                  # [inputs]
    basis: np.ndarray                 # [inputs, r]
    w: np.ndarray                     # [2r]
    b: float
    threshold: float
    meta: dict
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "MergeFilter":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        return cls(z["mean"].astype(np.float32), z["basis"].astype(np.float32), z["w"].astype(np.float32), float(z["b"]), float(meta["threshold"]), meta,
                   hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def keep(self, stacked: np.ndarray) -> np.ndarray:
        """stacked: writer rows [T, inputs] in order -> boolean mask of rows that end a reader token. The last row is always kept."""
        z = (np.asarray(stacked, dtype=np.float32) - self.mean) @ self.basis
        nxt = np.concatenate((z[1:], np.zeros_like(z[:1])))
        mask = np.concatenate((z, nxt), axis=1) @ self.w + self.b >= self.threshold
        mask[-1] = True
        return mask


@dataclass(frozen=True)
class MergeReader:
    """Reverse-direction reader with merging: writer tokens that do not end a reader token are dropped, and what they
    carried is folded into the merged entry through residual maps on top of the base translator (j-th predecessor)."""
    base: object                      # StackedReader with kv_heads == 0 (MLA latent reader)
    filter: MergeFilter
    basis: np.ndarray                 # [inputs, r]
    residual: dict                    # j -> [r, outputs] over all reader layers in base.reader_layers order
    meta: dict

    @classmethod
    def load(cls, path: Path, base, merge_filter: MergeFilter) -> "MergeReader":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        if meta.get("base_sha256") != base.sha256 or meta.get("filter_sha256") != merge_filter.sha256:
            raise ValueError("merge residuals were fitted for a different base translator or filter")
        return cls(base, merge_filter, z["basis"].astype(np.float32), {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("R")}, meta)

    def read(self, flat: dict, gain_power: float = 1.0, drop: bool = True) -> dict:
        """flat[writer layer] = [T, width] (K and V flattened). Returns {reader layer: latents [T', head_dim]}."""
        b = self.base
        x = np.concatenate([np.asarray(flat[l], dtype=np.float32) for l in b.writer_layers], axis=1)
        keep = self.filter.keep(np.concatenate([np.asarray(flat[l], dtype=np.float32) for l in self.filter.meta["layers"]], axis=1))
        x = x - b.input_mean
        z = x @ self.basis
        extra = np.zeros((len(x), next(iter(self.residual.values())).shape[1]), np.float32)
        for t in np.where(keep)[0]:
            j = 1
            while t - j >= 0 and not keep[t - j] and j in self.residual:
                extra[t] += z[t - j] @ self.residual[j]; j += 1
        rows = keep if drop else np.ones(len(x), bool)
        out, width = {}, b.head_dim
        for i, l in enumerate(b.reader_layers):
            y = (x @ b.weights[l] + extra[:, i * width:(i + 1) * width]) * b.gains[l] ** gain_power + b.biases[l]
            out[l] = y[rows]
        return out
