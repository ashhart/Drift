"""The clustered sign-flip test is verified against brute-force enumeration; the bootstrap keeps pairs and clusters intact."""
from __future__ import annotations
import itertools
import random
import numpy as np
import pytest
from drift.eval.paired_stats import clustered_bootstrap, passage_differences, sign_flip_p


def brute_force(differences):
    d = [x for x in differences if x != 0]
    if not d:
        return 1.0
    observed, hits = sum(d), 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        hits += sum(s * abs(x) for s, x in zip(signs, d)) >= observed
    return hits / 2 ** len(d)


def test_exact_sign_flip_matches_brute_force_enumeration():
    rng = random.Random(0)
    for _ in range(400):
        d = [rng.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(rng.randint(0, 14))]
        assert sign_flip_p(d) == pytest.approx(brute_force(d), abs=1e-15), d
    assert sign_flip_p([]) == 1.0 and sign_flip_p([0, 0]) == 1.0
    assert sign_flip_p([1]) == 0.5 and sign_flip_p([-1]) == 1.0 and sign_flip_p([2, 1]) == 0.25
    assert sign_flip_p([1] * 10) == pytest.approx(2 ** -10)                     # all ten improved: only one assignment is as extreme
    assert sign_flip_p([2] * 5) == pytest.approx(2 ** -5)                       # magnitudes matter only through the sum
    assert sign_flip_p([2, -1, -1]) == pytest.approx(brute_force([2, -1, -1]))  # ties in the sum are counted as >=
    assert sign_flip_p([1, 1, 1, -1] * 30) < 1e-6                               # large inputs stay exact (no enumeration)


def test_clustering_makes_the_test_more_conservative_than_treating_questions_as_independent():
    # 6 passages, both questions improve in each: 12 "independent" wins would give 2^-12, the 6 clusters give 2^-6
    outcomes = {f"p{i}": [(True, False), (True, False)] for i in range(6)}
    d = list(passage_differences(outcomes).values())
    assert d == [2] * 6 and sign_flip_p(d) == pytest.approx(2 ** -6)
    mixed = {"a": [(True, False), (False, True)], "b": [(True, True), (True, False)]}  # a win and a loss inside one passage cancel
    assert passage_differences(mixed) == {"a": 0, "b": 1}


def test_bootstrap_resamples_whole_passages_and_keeps_both_pipelines_paired():
    outcomes = {"easy": [(True, True), (True, True)], "fixed": [(True, False), (True, False)], "hard": [(False, False), (False, False)], "one": [(True, False)]}
    out = clustered_bootstrap(outcomes, draws=4000, seed=1)
    assert out["estimate"] == pytest.approx(3 / 7) and out["passages"] == 4 and out["questions"] == 7
    assert out["low"] >= 0.0                                                     # A never loses a question, so NO resample of paired passages can be negative
    assert out["low"] < out["estimate"] < out["high"] <= 1.0
    # an unpaired or question-level bootstrap could produce negative differences here; passage-paired resampling cannot
    same = {f"p{i}": [(bool(i % 2), bool(i % 2))] * 2 for i in range(20)}
    flat = clustered_bootstrap(same, draws=2000, seed=2)
    assert flat["estimate"] == 0.0 and flat["low"] == 0.0 and flat["high"] == 0.0   # identical pipelines: the interval collapses to zero
    wide = clustered_bootstrap({f"p{i}": [(True, False)] * 2 if i < 3 else [(False, True)] * 2 for i in range(6)}, draws=4000, seed=3)
    assert wide["low"] < 0 < wide["high"]
