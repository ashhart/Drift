from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def paired_bootstrap(active: list[float], baseline: list[float],
                     repetitions: int = 10000, seed: int = 73) -> dict:
    """Paired experimental-unit bootstrap. Pass one score PER independent secret/task.

    Multiple questions/seeds for a secret must first be grouped at secret level.
    This is a descriptive percentile interval, not a universal power guarantee.
    """
    a, b = np.asarray(active, dtype=float), np.asarray(baseline, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2 or repetitions < 100:
        raise ValueError("paired nontrivial samples and >=100 resamples required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("include failed trials explicitly; do not drop them as NaN")
    difference = a - b
    generator = np.random.default_rng(seed)
    values = []
    for start in range(0, repetitions, 256):
        rows = generator.integers(0, a.size, size=(min(256, repetitions - start), a.size))
        values.append(difference[rows].mean(axis=1))
    lower, upper = np.quantile(np.concatenate(values), [0.025, 0.975])
    return {"units": int(a.size), "active_mean": float(a.mean()), "baseline_mean": float(b.mean()),
            "paired_delta": float(difference.mean()), "ci95": [float(lower), float(upper)],
            "bootstrap_seed": seed, "repetitions": repetitions}


def e2_gate(summary: dict, *, valid_channel_audit: bool,
            preregistered: bool, minimum_units: int = 100) -> str:
    if not valid_channel_audit:
        return "INVALID"
    if not preregistered or summary["units"] < minimum_units:
        return "BLOCKED"
    return "PASSED" if summary["ci95"][0] > 0 else "FAILED"


@dataclass
class CostLedger:
    # All costs, including local token-conditioned mailbox writing, are declared.
    source_prefill_tokens: int = 0
    receiver_prefill_tokens: int = 0
    local_decode_tokens: int = 0
    mailbox_local_tokens: int = 0
    projector_calls: int = 0
    payload_bytes: int = 0
    wire_bytes: int = 0
    wall_seconds: float = 0.0
    joules: float | None = None       # None means UNMEASURED, never zero.

    def validate(self) -> None:
        integers = [self.source_prefill_tokens, self.receiver_prefill_tokens,
                    self.local_decode_tokens, self.mailbox_local_tokens,
                    self.projector_calls, self.payload_bytes, self.wire_bytes]
        if any(type(x) is not int or x < 0 for x in integers) or self.wall_seconds < 0:
            raise ValueError("negative/invalid accounting")
        if self.joules is not None and self.joules < 0:
            raise ValueError("invalid energy")
