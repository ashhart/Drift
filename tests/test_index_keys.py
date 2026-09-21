"""Selector keys are translated like K and V and follow fan-out row order."""
import json
import numpy as np
import pytest
from drift.translate.index_keys import IndexKeyReader


def test_index_key_reader_maps_rows_and_follows_fanout_order(tmp_path):
    rng = np.random.default_rng(0)
    W, b = rng.standard_normal((6, 8)).astype(np.float32), rng.standard_normal(8).astype(np.float32)
    path = tmp_path / "index.npz"
    np.savez(path, W=W, b=b, gain=np.full(8, 2.0, np.float32), meta=np.array(json.dumps({"index_dim": 4, "base_sha256": "ab" * 32})))
    reader = IndexKeyReader.load(path, layers=(3, 7), base_sha256="ab" * 32)
    x = rng.standard_normal((3, 6)).astype(np.float32)
    out = reader.read(x, order=np.array([0, 1, 1, 2]))
    assert set(out) == {3, 7} and out[7].shape == (4, 4)
    np.testing.assert_allclose(np.concatenate((out[3], out[7]), 1), ((x @ W) * 2.0 + b)[[0, 1, 1, 2]], rtol=1e-5)
    np.testing.assert_allclose(reader.read(x, gain_power=0.0)[3], (x @ W + b)[:, :4], rtol=1e-5)
    with pytest.raises(ValueError, match="different base"):
        IndexKeyReader.load(path, layers=(3, 7), base_sha256="cd" * 32)
    with pytest.raises(ValueError, match="declared reader layers"):
        IndexKeyReader.load(path, layers=(3, 7, 11), base_sha256="ab" * 32)
    with pytest.raises(ValueError, match="do not match"):
        reader.read(np.zeros((2, 5), np.float32))
