"""The spread gain undoes a translator's shrinkage toward its mean, per output dimension."""
import numpy as np
import pytest
from drift.translate.spread import restore, spread_gain


def test_gain_restores_each_dimensions_spread_about_the_prediction_mean():
    rng = np.random.default_rng(0)
    targets = rng.standard_normal((400, 3)) * np.array([1.0, 4.0, 0.5]) + np.array([2.0, -1.0, 0.0])
    predictions = 0.1 + np.array([0.5, 0.8, 0.25]) * targets                  # shrunk by a different factor per dimension
    centre, gain = spread_gain(predictions, targets)
    np.testing.assert_allclose(gain, [2.0, 1.25, 4.0], rtol=1e-5)
    np.testing.assert_allclose(centre, predictions.mean(0), rtol=1e-5)
    restored = restore(predictions, centre, gain)
    np.testing.assert_allclose(restored.std(0), targets.std(0), rtol=1e-5)
    np.testing.assert_allclose(restored.mean(0), predictions.mean(0), rtol=1e-5)      # the mean is left where the translator put it


def test_constant_dimension_does_not_divide_by_zero():
    predictions = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 4.0]])
    centre, gain = spread_gain(predictions, np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]))
    assert np.isfinite(gain).all() and gain[1] == pytest.approx(1.0)
    np.testing.assert_allclose(restore(predictions, centre, gain)[:, 0], 1.0)


def test_mismatched_or_single_row_inputs_are_refused():
    with pytest.raises(ValueError, match="matching"):
        spread_gain(np.zeros((4, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="at least two rows"):
        spread_gain(np.zeros((1, 3)), np.zeros((1, 3)))
