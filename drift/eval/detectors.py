"""Degeneration diagnostics (spec M2.3). Numbers, not verdicts; thresholds are tuned
on development data and frozen before evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence
import math
import torch
from drift.core.attention import Gate


def repetition_rate(tokens: Sequence[int], n: int = 4) -> float:
    """Fraction of n-grams that already occurred earlier in the sequence."""
    if len(tokens) < n + 1:
        return 0.0
    seen, repeated, total = set(), 0, 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        total += 1
        if gram in seen:
            repeated += 1
        seen.add(gram)
    return repeated / total


def gate_saturation(gates: Mapping[int, Gate], low: float = 1e-4, high: float = 0.99) -> dict[int, str]:
    out = {}
    for layer, gate in gates.items():
        g = float(torch.sigmoid(gate.logit.detach()))
        out[layer] = "closed" if g < low else "saturated" if g > high else "open"
    return out


@dataclass
class DriftTracker:
    """Per-layer running statistics of canonical-entry norms and foreign mass across epochs."""
    norms: dict[int, list[float]] = field(default_factory=dict)
    masses: dict[int, list[float]] = field(default_factory=dict)

    def observe(self, canonical_norms: Mapping[int, float], foreign_mass: Mapping[int, torch.Tensor]) -> None:
        for layer, value in canonical_norms.items():
            self.norms.setdefault(layer, []).append(float(value))
        for layer, mass in foreign_mass.items():
            self.masses.setdefault(layer, []).append(float(mass.detach().float().mean()))

    def report(self) -> dict:
        def drift(series: list[float]) -> float:
            if len(series) < 2 or series[0] == 0:
                return 0.0
            return (series[-1] - series[0]) / abs(series[0])

        def oscillation(series: list[float]) -> float:
            if len(series) < 3:
                return 0.0
            diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
            flips = sum(1 for a, b in zip(diffs, diffs[1:]) if a * b < 0)
            return flips / (len(diffs) - 1)

        return {
            "norm_drift": {layer: drift(s) for layer, s in self.norms.items()},
            "mass_mean": {layer: (sum(s) / len(s)) for layer, s in self.masses.items()},
            "mass_oscillation": {layer: oscillation(s) for layer, s in self.masses.items()},
            "nonfinite": any(not math.isfinite(v) for s in list(self.norms.values()) + list(self.masses.values()) for v in s),
        }
