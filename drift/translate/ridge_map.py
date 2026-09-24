"""Per-token translators written by scripts/live/fit_state_translator.py: load one and apply it to a sender's features.

A translator centres the sender's features, reduces them to principal directions, and maps them to each receiver layer
with a ridge weight, a gain and an output mean. With phases, token i uses phase i % phases, which has its own centre in
the reduced space, weight and output mean: DeepSeek V4 keeps one entry per four tokens, so its four tokens share features
and differ only in their phase. NumPy only.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load(path: Path, power: float = 1.0) -> dict:
    """The gain is raised to `power`; 1 restores the validation spread, 0 keeps ridge's shrunk outputs."""
    z = np.load(path)
    phases = int(z["phases"]) if "phases" in z.files else 1
    layers = {}
    for layer in (int(l) for l in z["layers"]):
        weight = z[f"W{layer}"].astype(np.float32)
        if weight.ndim != (3 if phases > 1 else 2) or (phases > 1 and (len(weight) != phases or len(z[f"b{layer}"]) != phases)):
            raise ValueError(f"layer {layer}: weight or bias shape does not match {phases} phases")
        layers[layer] = (weight, z[f"b{layer}"].astype(np.float32), z[f"gain{layer}"].astype(np.float32) ** power)
    centre = z["phase_mean"].astype(np.float32) if phases > 1 else np.zeros((1, z["basis"].shape[1]), np.float32)
    return {"mean": z["mean"].astype(np.float32), "basis": z["basis"].astype(np.float32), "phases": phases, "centre": centre, "layers": layers}


def apply(translator: dict, features: np.ndarray) -> dict[int, np.ndarray]:
    """Sender features [tokens, width] -> {receiver layer: [tokens, output width]}, in the sender's token order."""
    reduced = (np.asarray(features, np.float32) - translator["mean"]) @ translator["basis"]
    phases, out = translator["phases"], {}
    for layer, (weight, bias, gain) in translator["layers"].items():
        if phases == 1:
            out[layer] = (reduced @ weight) * gain + bias
            continue
        mapped = np.empty((len(reduced), weight.shape[2]), np.float32)
        for phase in range(phases):
            mapped[phase::phases] = ((reduced[phase::phases] - translator["centre"][phase]) @ weight[phase]) * gain + bias[phase]
        out[layer] = mapped
    return out


def fold(weights: np.ndarray, biases: np.ndarray, log_scale: np.ndarray, down: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A per-layer scale and a low-rank correction on top of a linear map, folded into one linear map per layer.

    weights [layers, rank_in, width] with the gain already in them, biases [layers, width], log_scale [layers],
    down [rank_in, rank], up [layers, rank, width]: (z @ W + b) * exp(s) + (z @ down) @ up == z @ W' + b'."""
    scale = np.exp(np.asarray(log_scale, np.float32))
    return weights * scale[:, None, None] + np.einsum("ir,lro->lio", down, up), biases * scale[:, None]

