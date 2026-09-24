"""A shared space for cache translation: each model maps into one hub and out of it, so N models need 2N maps, not N^2.

The hub is the space of an anchor model's reduced per-token features (GLM-5.3's MLA latents in 2,048 principal
directions here). A member is one model: an encoder from its per-token features into the hub, and a decoder from the
hub to the per-token rows it can take into its cache. For the anchor, both are the principal-direction projection and
its inverse. A translator from one member to another is the source's encoder followed by the target's decoder; both
are linear, so the pair folds into one ridge_map translator (drift/translate/ridge_map.py) that the existing gates and
drop-in tools load unchanged. Fitting, evaluation and the drop-in test are in scripts/live/hub_add_model.py. NumPy only.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Member:
    """One model in the shared space. Encoder: z = ((x - in_mean) @ in_basis) @ enc_w + enc_b. Decoder: y = (z @ dec_w + dec_b)
    rescaled about dec_centre by dec_gain; the anchor's encoder is its projection (enc_w the identity, enc_b zero)."""
    name: str
    in_mean: np.ndarray
    in_basis: np.ndarray
    enc_w: np.ndarray
    enc_b: np.ndarray
    dec_w: np.ndarray
    dec_b: np.ndarray
    dec_gain: np.ndarray
    dec_centre: np.ndarray

    def encode(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, np.float32) - self.in_mean) @ self.in_basis) @ self.enc_w + self.enc_b

    def decode(self, z: np.ndarray) -> np.ndarray:
        y = np.asarray(z, np.float32) @ self.dec_w + self.dec_b
        return self.dec_centre + self.dec_gain * (y - self.dec_centre)


def anchor(name: str, mean: np.ndarray, basis: np.ndarray) -> Member:
    """The hub's own model: the projection into its principal directions, and back."""
    mean, basis = np.asarray(mean, np.float32), np.asarray(basis, np.float32)
    rank, width = basis.shape[1], basis.shape[0]
    return Member(name, mean, basis, np.eye(rank, dtype=np.float32), np.zeros(rank, np.float32),
                  basis.T.copy(), mean.copy(), np.ones(width, np.float32), mean.copy())


def save(member: Member, folder: Path, report: dict | None = None) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "member.npz", **{k: getattr(member, k) for k in ("in_mean", "in_basis", "enc_w", "enc_b", "dec_w", "dec_b", "dec_gain", "dec_centre")})
    (folder / "member.json").write_text(json.dumps({"name": member.name, "hub_rank": int(member.enc_w.shape[1]),
                                                    "feature_width": int(member.in_basis.shape[0]), "row_width": int(member.dec_w.shape[1]),
                                                    **(report or {})}, indent=1) + "\n")


def load(folder: Path) -> Member:
    folder = Path(folder)
    arrays = np.load(folder / "member.npz")
    name = json.loads((folder / "member.json").read_text())["name"]
    return Member(name, **{k: arrays[k].astype(np.float32) for k in arrays.files})


def pair(source: Member, target: Member) -> dict:
    """source -> target as one ridge_map translator: its mean and basis, and one layer of weight, bias and gain.

    y = centre + gain * (((x - mean) @ basis @ enc_w + enc_b) @ dec_w + dec_b - centre), which is ridge_map's
    ((x - mean) @ basis) @ W * gain + b with W = enc_w @ dec_w and b = centre + gain * (enc_b @ dec_w + dec_b - centre)."""
    if source.enc_w.shape[1] != target.dec_w.shape[0]:
        raise ValueError("the two members do not share one hub")
    weight = source.enc_w @ target.dec_w
    bias = target.dec_centre + target.dec_gain * (source.enc_b @ target.dec_w + target.dec_b - target.dec_centre)
    return {"mean": source.in_mean, "basis": source.in_basis, "phases": 1, "centre": np.zeros((1, source.in_basis.shape[1]), np.float32),
            "layers": {0: (weight.astype(np.float32), bias.astype(np.float32), target.dec_gain.astype(np.float32))}}


def save_translator(translator: dict, path: Path) -> None:
    """A paired translator in the npz layout ridge_map.load reads."""
    (weight, bias, gain), = translator["layers"].values()
    np.savez(path, mean=translator["mean"], basis=translator["basis"], layers=np.array([0], np.int32), W0=weight, b0=bias, gain0=gain)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float, chunk: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an L2 penalty scaled by the features' variance: y ~ x @ w + b. Sums in float64 a chunk at a time."""
    xm, ym = np.asarray(x).mean(0, dtype=np.float64), np.asarray(y).mean(0, dtype=np.float64)
    gram, cross = np.zeros((x.shape[1], x.shape[1])), np.zeros((x.shape[1], y.shape[1]))
    for start in range(0, len(x), chunk):
        xc = np.asarray(x[start:start + chunk], np.float64) - xm
        gram += xc.T @ xc
        cross += xc.T @ (np.asarray(y[start:start + chunk], np.float64) - ym)
    gram[np.diag_indices_from(gram)] += ridge * np.trace(gram) / len(gram)
    w = np.linalg.solve(gram, cross)
    return w.astype(np.float32), (ym - xm @ w).astype(np.float32)


def principal(x: np.ndarray, rank: int, chunk: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Mean and the top principal directions [width, rank] of x's rows, from their covariance, a chunk at a time."""
    mean = np.asarray(x).mean(0, dtype=np.float64)
    gram = np.zeros((x.shape[1], x.shape[1]))
    for start in range(0, len(x), chunk):
        xc = np.asarray(x[start:start + chunk], np.float64) - mean
        gram += xc.T @ xc
    _, vectors = np.linalg.eigh(gram)
    return mean.astype(np.float32), np.ascontiguousarray(vectors[:, ::-1][:, :rank]).astype(np.float32)
