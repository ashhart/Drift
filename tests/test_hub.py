"""Members of the shared space compose into one ridge_map translator that equals encoding then decoding."""
import numpy as np
from drift.translate import hub, ridge_map


def _member(rng, name, width, rank, rows):
    mean, basis = rng.standard_normal(width).astype(np.float32), np.linalg.qr(rng.standard_normal((width, rank)))[0].astype(np.float32)
    return hub.Member(name, mean, basis, rng.standard_normal((rank, 4)).astype(np.float32), rng.standard_normal(4).astype(np.float32),
                      rng.standard_normal((4, rows)).astype(np.float32), rng.standard_normal(rows).astype(np.float32),
                      rng.uniform(0.5, 2, rows).astype(np.float32), rng.standard_normal(rows).astype(np.float32))


def test_a_pair_folds_into_one_translator_that_matches_encode_then_decode(tmp_path):
    rng = np.random.default_rng(0)
    source, target = _member(rng, "a", 9, 5, 7), _member(rng, "b", 6, 3, 11)
    x = rng.standard_normal((13, 9)).astype(np.float32)
    folded = hub.pair(source, target)
    np.testing.assert_allclose(ridge_map.apply(folded, x)[0], target.decode(source.encode(x)), rtol=1e-4, atol=1e-4)
    hub.save_translator(folded, tmp_path / "pair.npz")
    np.testing.assert_allclose(ridge_map.apply(ridge_map.load(tmp_path / "pair.npz"), x)[0], target.decode(source.encode(x)), rtol=1e-4, atol=1e-4)


def test_the_anchor_projects_into_the_hub_and_back_and_members_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    width, rank = 8, 8
    mean, basis = hub.principal(rng.standard_normal((50, width)), rank)
    a = hub.anchor("glm", mean, basis)
    x = rng.standard_normal((5, width)).astype(np.float32)
    np.testing.assert_allclose(a.decode(a.encode(x)), x, rtol=1e-4, atol=1e-4)        # full rank: the projection loses nothing
    hub.save(a, tmp_path / "glm", {"note": "anchor"})
    again = hub.load(tmp_path / "glm")
    assert again.name == "glm"
    np.testing.assert_allclose(again.encode(x), a.encode(x), rtol=1e-5, atol=1e-5)


def test_ridge_recovers_a_linear_map_and_members_of_different_hubs_are_refused():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((400, 6))
    w_true, b_true = rng.standard_normal((6, 3)), rng.standard_normal(3)
    w, b = hub.fit_ridge(x, x @ w_true + b_true, ridge=1e-6)
    np.testing.assert_allclose(w, w_true, atol=1e-3)
    np.testing.assert_allclose(b, b_true, atol=1e-3)
    try:
        hub.pair(_member(rng, "a", 9, 5, 7), hub.Member("c", *(np.zeros(s, np.float32) for s in ((6,), (6, 3), (3, 2), (2,), (5, 7), (7,), (7,), (7,)))))
    except ValueError as error:
        assert "hub" in str(error)
    else:
        raise AssertionError("members of different hubs paired")
