"""Paired windows for a contextual reader: GLM's latents and Qwen's own rows for the same text, either way. NumPy only.

GLM exports carry token offsets (export_passages.py); Qwen taps carry offsets and the rows x (studio_tap_passages.py);
each window is one npz per side under the same name. A GLM position is paired with Qwen's row for the token that ends
on the same character (drift/translate/alignment.py).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
from drift.translate.alignment import aligned_pairs


def layer_features(export) -> np.ndarray:
    """GLM's per-layer latents in one export, concatenated in layer order: [tokens, layers x width], float32."""
    keys = sorted((k for k in export.files if k[0] == "l" and k[1:].isdigit()), key=lambda k: int(k[1:]))
    if not keys:
        raise ValueError("the export holds no per-layer latents")
    return np.concatenate([export[k] for k in keys], axis=1).astype(np.float32)


def windows(glm_dir: Path, qwen_dir: Path, min_pairs: int = 16) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(name, GLM features, aligned GLM positions, Qwen's rows at them) for each window both sides exported exactly."""
    for path in sorted(Path(glm_dir).glob("*.npz")):
        other = Path(qwen_dir) / path.name
        if not other.exists():
            continue
        glm = np.load(path)
        if "aligned" in glm.files and not bool(glm["aligned"]):
            continue                                                    # inexact offsets would pair the wrong tokens
        qwen = np.load(other)
        source, target = aligned_pairs(glm["offsets"], qwen["offsets"])
        if len(source) < min_pairs:
            continue
        yield path.stem, layer_features(glm), source, qwen["x"][target]


def reverse_windows(glm_dir: Path, qwen_dir: Path, min_pairs: int = 16) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """The same windows read the other way: (name, Qwen's rows, aligned Qwen positions, GLM's features at them)."""
    for path in sorted(Path(glm_dir).glob("*.npz")):
        other = Path(qwen_dir) / path.name
        if not other.exists():
            continue
        glm = np.load(path)
        if "aligned" in glm.files and not bool(glm["aligned"]):
            continue
        qwen = np.load(other)
        target, source = aligned_pairs(glm["offsets"], qwen["offsets"])
        if len(source) < min_pairs:
            continue
        yield path.stem, qwen["x"].astype(np.float32), source, layer_features(glm)[target]

