"""Give translated rows back the spread that squared-error training takes out of them. NumPy only.

A translator trained on squared error pulls its outputs toward their mean: that lowers the error, but the rows it writes
are flatter than the receiver's own, so attention over them is flatter too. The gain rescales each output dimension
about the predictions' mean so its spread matches the targets'. It is fitted once on paired predictions and targets and
applied to every later prediction.
"""
from __future__ import annotations
import numpy as np


def spread_gain(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per output dimension, the predictions' mean and the gain that gives them the targets' standard deviation."""
    predictions, targets = np.asarray(predictions, np.float64), np.asarray(targets, np.float64)
    if predictions.ndim != 2 or predictions.shape != targets.shape or len(predictions) < 2:
        raise ValueError("predictions and targets must be matching [rows, width] arrays with at least two rows")
    gain = targets.std(0) / np.maximum(predictions.std(0), eps)
    return predictions.mean(0).astype(np.float32), gain.astype(np.float32)


def restore(rows: np.ndarray, centre: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Rows [tokens, width] rescaled about the centre by the gain."""
    return centre + gain * (rows - centre)
