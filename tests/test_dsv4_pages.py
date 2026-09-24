"""DeepSeek V4 page codec: entries survive a page round trip, scales sit after all data, misaligned rows are refused."""
import numpy as np
import pytest
from drift.serving.glm53_handoff import E4M3FN
from drift.translate.dsv4_pages import DATA_BYTES, ENTRY_BYTES, INDEX_BYTES, encode_entries, encode_index_keys, entries, index_keys


def sample(n, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 512)).astype(np.float32) * np.exp2(rng.integers(-6, 6, size=(n, 1))).astype(np.float32)
    rope = values[:, 448:].view(np.uint32) & 0xFFFF0000                     # rope values exact in bfloat16
    values[:, 448:] = rope.view(np.float32)
    return values


@pytest.mark.parametrize("per_page", [64, 2])
def test_entries_survive_a_page_round_trip(per_page):
    values = sample(per_page * 3)
    rows = encode_entries(values, per_page)
    assert rows.shape == (len(values), ENTRY_BYTES) and rows.dtype == np.uint8
    back = entries(rows, per_page)
    assert np.array_equal(back[:, 448:], values[:, 448:])
    blocks = np.abs(values[:, :448]).reshape(len(values), 7, 64).max(-1, keepdims=True)
    error = np.abs(back[:, :448] - values[:, :448]).reshape(len(values), 7, 64)
    assert (error <= blocks / 16 + 1e-6).all()


def test_scales_follow_every_entry_data_in_a_page():
    values = np.zeros((2, 512), np.float32)
    values[0, :64], values[1, 64:128] = 448.0 * 4, 448.0 / 2
    page = encode_entries(values, 2).reshape(-1)
    scales = page[2 * DATA_BYTES:].reshape(2, 8)
    assert scales[0, 0] == 127 + 2 and scales[1, 1] == 127 - 1 and not scales[:, 7].any()
    assert E4M3FN[page[0]] * 4 == 448.0 * 4 and E4M3FN[page[DATA_BYTES + 64]] / 2 == 448.0 / 2


def test_rows_that_are_not_whole_aligned_pages_are_refused():
    rows = encode_entries(sample(128, seed=1), 64)
    with pytest.raises(ValueError, match="whole pages"):
        entries(rows[:63], 64)
    with pytest.raises(ValueError, match="pad bytes"):
        entries(np.roll(rows, 1, axis=0), 64)


def test_indexer_keys_are_codes_times_their_entry_scale():
    codes = np.arange(4 * 128, dtype=np.int64).reshape(4, 128) % 0x7E
    scales = np.array([0.5, 1.0, 2.0, 4.0], "<f4")
    page = np.concatenate((codes.astype(np.uint8).reshape(-1), scales.view(np.uint8)))
    keys = index_keys(page.reshape(4, 132), 4)
    assert np.array_equal(keys, E4M3FN[codes] * scales[:, None])
    broken = page.copy()
    broken[4 * 128:] = 0
    with pytest.raises(ValueError, match="aligned pages"):
        index_keys(broken.reshape(4, 132), 4)


def test_indexer_keys_survive_a_page_round_trip():
    keys = sample(8, seed=2)[:, :128]
    rows = encode_index_keys(keys, 4)
    assert rows.shape == (8, INDEX_BYTES) and rows.dtype == np.uint8
    back = index_keys(rows, 4)
    peak = np.abs(keys).max(1, keepdims=True)
    assert (np.abs(back - keys) <= peak / 16 + 1e-6).all()
    with pytest.raises(ValueError, match="whole pages"):
        encode_index_keys(keys[:7], 4)

