"""Sparse residual artifacts must clamp to complete emission prefixes."""
import numpy as np
import pytest

from drift.translate.fanout import FanoutReader
from test_fanout_merge import base_reader


@pytest.mark.parametrize("keys,expected", [([], 1), ([3], 1), ([1, 3], 2), ([1, 2, 3], 4)])
def test_missing_residuals_cannot_be_selected(tmp_path, keys, expected):
    base = base_reader(tmp_path, np.random.default_rng(11))
    fan = FanoutReader(base, np.eye(6, dtype=np.float32), np.zeros((6, 4), np.float32),
                       np.array([0, 1, 2, 10], np.float32), 0,
                       {key: np.ones((6, 8), np.float32) for key in keys}, {}, "synthetic")
    rows = np.zeros((1, 6), np.float32)
    assert fan.spans(rows).tolist() == [expected]
    assert fan.read({3: rows[:, :3], 7: rows[:, 3:]})[1][0].shape[0] == expected


def test_mlx_read_algorithm_uses_the_same_clamp_with_a_cpu_array_standin(tmp_path):
    from types import SimpleNamespace as NS
    from drift.translate.mlx_reader import MlxForwardReader
    base = base_reader(tmp_path, np.random.default_rng(13))
    residual = {key: np.ones((6, 8), np.float32) for key in (1, 3)}
    fan = FanoutReader(base, np.eye(6, dtype=np.float32), np.zeros((6, 4), np.float32),
                       np.array([0, 1, 2, 10], np.float32), 0, residual, {}, "synthetic")
    reader = object.__new__(MlxForwardReader)
    reader.mx = NS(array=np.asarray, float16=np.float16, eval=lambda *args: None)
    reader.reader = NS(fan=fan)
    reader.layers, reader.heads, reader.dim = (1,), 2, 2
    reader.mean, reader.basis, reader.residual = base.input_mean, fan.basis, residual
    reader.W, reader.A, reader.B = base.weights[1], np.zeros((6, 1)), np.zeros((1, 8))
    reader.scale, reader.bias, reader.index = base.gains[1], base.biases[1], None
    entries, _, order, which = reader.read({3: np.zeros((1, 3)), 7: np.zeros((1, 3))})
    assert order.tolist() == [0, 0] and which.tolist() == [1, 0]
    assert entries[1][0].shape == (2, 2, 2)
