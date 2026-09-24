"""DeepSeek's shared-space decoder: entries pool the right tokens, and injection replaces only translated content."""
import numpy as np
from drift.translate import dsv4_member
from drift.translate.dsv4_pages import encode_entries, encode_index_keys


def _span_rows(rng, tokens=256):
    """A placeholder span's raw rows for two ratio-4 layers with indexers and one ratio-128 layer."""
    rows = {}
    for layer in (2, 5):
        values = rng.normal(size=(tokens // 4, 512)).astype(np.float32)
        values[:, 448:] = (values[:, 448:].view(np.uint32) & 0xFFFF0000).view(np.float32)
        rows[f"model.layers.{layer}.attn"] = (4, encode_entries(values, 64))
        rows[f"model.layers.{layer}.attn.indexer.k_cache"] = (4, encode_index_keys(rng.normal(size=(tokens // 4, 128)), 64))
    values = rng.normal(size=(tokens // 128, 512)).astype(np.float32)
    values[:, 448:] = (values[:, 448:].view(np.uint32) & 0xFFFF0000).view(np.float32)
    rows["model.layers.3.attn"] = (128, encode_entries(values, 2))
    return rows


def test_entries_pool_the_tokens_that_end_inside_them():
    offsets = np.array([[0, 3], [3, 5], [5, 9], [9, 10], [10, 14], [14, 15]])      # a receiver's six passage tokens
    whole = dsv4_member.spans(offsets, 6, 4, 64, whole=True)
    assert whole[0] == (0, 10) and whole[1] is None and whole[2] is None
    partial = dsv4_member.spans(offsets, 6, 4, 64, whole=False)
    assert partial[1] == (10, 15) and partial[2] is None
    source = np.array([[0, 5], [5, 10], [10, 12], [12, 15]])                        # a sender's four tokens
    z = np.arange(8, dtype=np.float32).reshape(4, 2)
    pooled, kept = dsv4_member.pool(z, source, partial)
    assert list(kept) == [0, 1]
    np.testing.assert_allclose(pooled, [[1, 2], [5, 6]])


def test_injection_replaces_translated_content_and_keeps_rope_and_other_entries():
    rng = np.random.default_rng(0)
    blank = _span_rows(rng)
    before = dsv4_member.content(blank)
    kept4, kept128 = np.array([0, 1, 7]), np.array([0])
    mla4, index4 = rng.normal(size=(3, 2, 448)).astype(np.float32), rng.normal(size=(3, 2, 128)).astype(np.float32)
    mla128 = rng.normal(size=(1, 1, 448)).astype(np.float32)
    after_rows = dsv4_member.inject_rows(blank, mla4, index4, kept4, mla128, kept128)
    after = dsv4_member.content(after_rows)
    np.testing.assert_allclose(after["mla4"][kept4], mla4, atol=0.2)
    np.testing.assert_allclose(after["index4"][kept4], index4, atol=0.2)
    np.testing.assert_allclose(after["mla128"][kept128], mla128, atol=0.2)
    untouched = np.setdiff1d(np.arange(64), kept4)
    np.testing.assert_allclose(after["mla4"][untouched], before["mla4"][untouched], atol=1e-6)
    from drift.translate.dsv4_pages import entries
    for name in ("model.layers.2.attn", "model.layers.5.attn"):
        np.testing.assert_array_equal(entries(after_rows[name][1], 64)[:, 448:], entries(blank[name][1], 64)[:, 448:])


def test_the_decoder_shapes_its_output_by_layer_and_survives_saving(tmp_path):
    rng = np.random.default_rng(1)
    decoder = dsv4_member.Decoder(np.array([2, 5]), np.array([3]), rng.normal(size=(6, 2 * 576)).astype(np.float32), np.zeros(2 * 576, np.float32),
                                  np.ones(2 * 576, np.float32), np.zeros(2 * 576, np.float32), rng.normal(size=(6, 448)).astype(np.float32),
                                  np.zeros(448, np.float32), np.ones(448, np.float32), np.zeros(448, np.float32))
    pooled = rng.normal(size=(5, 6)).astype(np.float32)
    mla4, index4 = decoder.ratio4(pooled)
    assert mla4.shape == (5, 2, 448) and index4.shape == (5, 2, 128) and decoder.ratio128(pooled).shape == (5, 1, 448)
    np.testing.assert_allclose(index4[:, 1], (pooled @ decoder.w4)[:, 576 + 448:])
    dsv4_member.save(decoder, tmp_path / "dsv4", {"note": "test"})
    np.testing.assert_allclose(dsv4_member.load(tmp_path / "dsv4").ratio128(pooled), decoder.ratio128(pooled))


def test_slots_put_each_receiver_token_s_sender_vector_side_by_side():
    offsets = np.array([[0, 3], [3, 5], [5, 9], [9, 10], [10, 14]])                # five receiver tokens: one whole entry, one partial
    source = np.array([[0, 5], [5, 10], [10, 14]])
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    x, kept = dsv4_member.slots(z, source, offsets, 5, 4, 64, 2)
    assert list(kept) == [0, 1] and x.shape == (2, 8)
    np.testing.assert_array_equal(x[0], [0, 1, 0, 1, 3, 4, 3, 4])                 # tokens ending at 3 and 5 sit under the first sender token
    np.testing.assert_array_equal(x[1], [6, 7, 0, 0, 0, 0, 0, 0])

