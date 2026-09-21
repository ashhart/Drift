"""Diagnostic probe training (spec M3.2, D9): a linear probe over canonical entries predicts
labels on development data; report held-out accuracy and calibration. Probe outputs are
experimenter-only and are never returned to any worker."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Sequence
import torch
from torch.nn import functional as F
from drift.core.types import KV
from drift.mailbox.protocol import ProbeHead


@dataclass(frozen=True)
class ProbeExample:
    entry: KV
    label: int


def features(entry: KV) -> torch.Tensor:
    return torch.cat((entry.k.detach().mean(0).flatten(), entry.v.detach().mean(0).flatten()))


def train_probe(probe: ProbeHead, train: Sequence[ProbeExample], heldout: Sequence[ProbeExample],
                steps: int = 200, learning_rate: float = 1e-2, seed: int = 0) -> dict:
    if not train or not heldout or steps <= 0:
        raise ValueError("training and held-out examples and a positive step budget are required")
    labels = probe.head.out_features
    if any(not 0 <= e.label < labels for e in list(train) + list(heldout)):
        raise ValueError("label outside the probe's label set")
    torch.manual_seed(seed)
    x = torch.stack([features(e.entry) for e in train])
    y = torch.tensor([e.label for e in train])
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    started = time.perf_counter()
    for _ in range(steps):
        loss = F.cross_entropy(probe.head(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        hx = torch.stack([features(e.entry) for e in heldout])
        hy = torch.tensor([e.label for e in heldout])
        probs = probe.head(hx).softmax(-1)
        confidence, predicted = probs.max(-1)
        correct = (predicted == hy).float()
        # Expected calibration error over 10 confidence bins.
        bins = torch.clamp((confidence * 10).long(), max=9)
        ece = 0.0
        for b in range(10):
            mask = bins == b
            if mask.any():
                ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    return {"train_examples": len(train), "heldout_examples": len(heldout), "steps": steps,
            "heldout_accuracy": float(correct.mean()), "mean_confidence": float(confidence.mean()),
            "expected_calibration_error": ece, "chance": 1.0 / labels,
            "wall_seconds": time.perf_counter() - started, "audience": "experimenter_only"}
