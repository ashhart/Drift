"""Windows pair GLM positions with Qwen's own rows only where both tokenizations end on the same character."""
import numpy as np
import pytest
from drift.translate.context_pairs import layer_features, reverse_windows, windows


def _glm(path, tokens, offsets, aligned=None):
    arrays = {f"l{layer}": np.full((tokens, 2), layer, np.float16) for layer in (11, 3, 7)}
    if aligned is not None:
        arrays["aligned"] = np.array(aligned)
    np.savez(path, offsets=np.asarray(offsets), **arrays)


def test_layers_concatenate_in_numeric_order_not_text_order(tmp_path):
    _glm(tmp_path / "a.npz", 2, [[0, 1], [1, 2]])
    features = layer_features(np.load(tmp_path / "a.npz"))
    assert features.dtype == np.float32 and features[0].tolist() == [3, 3, 7, 7, 11, 11]
    np.savez(tmp_path / "empty.npz", offsets=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="no per-layer latents"):
        layer_features(np.load(tmp_path / "empty.npz"))


def test_windows_pair_shared_ends_and_skip_unusable_windows(tmp_path):
    glm, qwen = tmp_path / "glm", tmp_path / "qwen"
    glm.mkdir(); qwen.mkdir()
    offsets = [[i, i + 1] for i in range(20)]
    glm_offsets = offsets[:10] + [[10, 12]] + offsets[12:]                  # GLM joins characters 10 and 11 into one token
    _glm(glm / "good.npz", 19, glm_offsets)
    np.savez(qwen / "good.npz", offsets=np.asarray(offsets), x=np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32))
    _glm(glm / "flagged.npz", 20, offsets, aligned=False)
    np.savez(qwen / "flagged.npz", offsets=np.asarray(offsets), x=np.zeros((20, 4), np.float32))
    _glm(glm / "short.npz", 3, offsets[:3])
    np.savez(qwen / "short.npz", offsets=np.asarray(offsets[:3]), x=np.zeros((3, 4), np.float32))
    _glm(glm / "alone.npz", 20, offsets)
    found = list(windows(glm, qwen))
    assert [name for name, *_ in found] == ["good"]
    _, features, source, rows = found[0]
    assert features.shape == (19, 6)
    assert 10 in source.tolist() and len(source) == 19                      # the joined token pairs with Qwen's token ending on character 12
    assert rows[source.tolist().index(10), 0] == 11.0 and rows[source.tolist().index(9), 0] == 9.0


def test_reverse_windows_read_qwen_rows_and_target_glm_features_at_the_same_ends(tmp_path):
    glm, qwen = tmp_path / "glm", tmp_path / "qwen"
    glm.mkdir(); qwen.mkdir()
    offsets = [[i, i + 1] for i in range(20)]
    glm_offsets = offsets[:10] + [[10, 12]] + offsets[12:]
    arrays = {f"l{layer}": np.arange(19, dtype=np.float16)[:, None] * np.ones((1, 2), np.float16) for layer in (3, 7)}
    np.savez(glm / "good.npz", offsets=np.asarray(glm_offsets), **arrays)
    np.savez(qwen / "good.npz", offsets=np.asarray(offsets), x=np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32))
    (name, rows, source, target), = reverse_windows(glm, qwen)
    assert name == "good" and rows.shape == (20, 4) and target.shape == (19, 4)
    assert 11 in source.tolist() and 10 not in source.tolist()               # Qwen's token ending on character 12 pairs with GLM's joined token
    assert target[source.tolist().index(11), 0] == 10.0 and target[source.tolist().index(9), 0] == 9.0

