"""Qwen rows round-trip between the flat tap layout and per-layer K/V entries."""
import numpy as np
import pytest
from drift.translate.qwen_rows import join_rows, split_rows, zero_selector_keys


def test_split_places_each_layers_k_then_v_and_join_inverts_it():
    layers = (3, 7)
    x = np.arange(3 * 2 * 1024, dtype=np.float32).reshape(3, 2048)
    entries = split_rows(x, layers)
    assert set(entries) == {3, 7} and entries[3][0].shape == (3, 2, 256)
    assert entries[3][0][0, 0, 0] == 0 and entries[3][1][0, 0, 0] == 512 and entries[7][0][0, 0, 0] == 1024 and entries[7][1][0, 1, 255] == 2047
    np.testing.assert_array_equal(join_rows(entries, layers), x)


def test_wrong_width_is_refused_and_zero_keys_cover_every_layer():
    with pytest.raises(ValueError, match="rows must be"):
        split_rows(np.zeros((2, 1000)), (3,))
    keys = zero_selector_keys(4, (3, 7), 128)
    assert set(keys) == {3, 7} and keys[7].shape == (4, 128) and not keys[3].any()
