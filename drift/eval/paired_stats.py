"""Paired, passage-clustered statistics for comparing two pipelines on the same questions (Test A, v4 vs v3).

Several questions share one passage and therefore one transferred memory, so the PASSAGE is the unit:
  d_p = (# questions of passage p that pipeline A answers correctly) - (# that pipeline B answers correctly).
Null hypothesis: within a passage the labels A / B are exchangeable, so each d_p keeps its magnitude and takes either sign
with probability 1/2. `sign_flip_p` is the EXACT one-sided p-value of sum(d_p) under that null, computed by convolving
the per-passage two-point distributions (no sampling, no normal approximation). `clustered_bootstrap` resamples whole
passages and keeps both pipelines' outcomes of a passage together."""
from __future__ import annotations
from fractions import Fraction
from typing import Mapping, Sequence
import numpy as np


def passage_differences(outcomes: Mapping[str, Sequence[tuple[bool, bool]]]) -> dict[str, int]:
    """outcomes[passage] = [(a_correct, b_correct), ...] for its questions -> {passage: d_p}."""
    return {p: sum(int(a) - int(b) for a, b in pairs) for p, pairs in outcomes.items()}


def sign_flip_p(differences: Sequence[int]) -> float:
    """Exact one-sided P(sum of sign-flipped differences >= observed sum). Zeros carry no information and drop out."""
    d = [int(x) for x in differences if int(x) != 0]
    if not d:
        return 1.0
    observed, span = sum(d), sum(abs(x) for x in d)
    counts = {0: 1}                                               # number of sign assignments reaching each partial sum (exact integers)
    for x in d:
        nxt: dict[int, int] = {}
        for total, ways in counts.items():
            for s in (total + abs(x), total - abs(x)):
                nxt[s] = nxt.get(s, 0) + ways
        counts = nxt
    assert -span <= observed <= span
    favourable = sum(ways for total, ways in counts.items() if total >= observed)
    return float(Fraction(favourable, 2 ** len(d)))


def clustered_bootstrap(outcomes: Mapping[str, Sequence[tuple[bool, bool]]], draws: int = 10000, seed: int = 0, level: float = 0.95) -> dict:
    """Difference in accuracy (A - B) with a percentile interval from resampling PASSAGES with replacement.
    A drawn passage contributes all of its questions, with A's and B's outcomes kept paired."""
    passages = list(outcomes)
    a = np.array([sum(int(x) for x, _ in outcomes[p]) for p in passages], dtype=np.float64)
    b = np.array([sum(int(y) for _, y in outcomes[p]) for p in passages], dtype=np.float64)
    n = np.array([len(outcomes[p]) for p in passages], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(passages), size=(draws, len(passages)))
    diffs = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)        # the same resampled passages feed both pipelines
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": float((a.sum() - b.sum()) / n.sum()), "low": float(lo), "high": float(hi), "level": level, "passages": len(passages), "questions": int(n.sum()), "draws": draws}
