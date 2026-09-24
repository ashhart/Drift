"""Per-token translators: loading, applying, and folding a trained correction back into one linear map."""
import numpy as np
from drift.translate.ridge_map import apply, fold, load


def test_a_folded_correction_is_the_same_map(tmp_path):
    rng = np.random.default_rng(0)
    layers, rank_in, rank, width = 3, 8, 2, 5
    weights, biases = rng.normal(size=(layers, rank_in, width)), rng.normal(size=(layers, width))
    log_scale, down, up = rng.normal(size=layers) * 0.1, rng.normal(size=(rank_in, rank)), rng.normal(size=(layers, rank, width))
    folded_w, folded_b = fold(weights, biases, log_scale, down, up)
    z = rng.normal(size=(4, rank_in))
    for j in range(layers):
        expected = (z @ weights[j] + biases[j]) * np.exp(log_scale[j]) + (z @ down) @ up[j]
        np.testing.assert_allclose(z @ folded_w[j] + folded_b[j], expected, rtol=1e-5, atol=1e-6)


def test_a_saved_translator_applies_its_gain_and_power(tmp_path):
    rng = np.random.default_rng(1)
    mean, basis = rng.normal(size=6).astype(np.float32), rng.normal(size=(6, 4)).astype(np.float32)
    weight, bias, gain = rng.normal(size=(4, 3)), rng.normal(size=3), np.array([1.0, 2.0, 4.0])
    np.savez(tmp_path / "t.npz", mean=mean, basis=basis, layers=np.array([7], np.int32), W7=weight, b7=bias, gain7=gain)
    x = rng.normal(size=(2, 6)).astype(np.float32)
    reduced = (x - mean) @ basis
    np.testing.assert_allclose(apply(load(tmp_path / "t.npz"), x)[7], (reduced @ weight) * gain + bias, rtol=1e-5)
    np.testing.assert_allclose(apply(load(tmp_path / "t.npz", power=0.5), x)[7], (reduced @ weight) * np.sqrt(gain) + bias, rtol=1e-5)
