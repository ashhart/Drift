"""A saved contextual reader reads rows by the trainer's formula, for either input, with and without the spread gain."""
import json
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def _folder(tmp_path, input_kind):
    from drift.translate.context_reader import ContextReader
    rng = np.random.default_rng(3)
    D, r, out = 6, 4, 8
    np.savez(tmp_path / "rows.npz", mean=rng.standard_normal(D).astype(np.float32), basis=np.linalg.qr(rng.standard_normal((D, r)))[0].astype(np.float32),
             layers=np.array([0], np.int32), W0=rng.standard_normal((r, out)).astype(np.float32), b0=rng.standard_normal(out).astype(np.float32),
             gain0=np.full(out, 1.5, np.float32))
    in_dim = D if input_kind == "full" else r
    np.savez(tmp_path / "scales.npz", component=rng.uniform(0.5, 2.0, in_dim).astype(np.float32), target=rng.uniform(0.5, 2.0, out).astype(np.float32),
             input_mean=rng.standard_normal(in_dim).astype(np.float32) if input_kind == "full" else np.zeros(in_dim, np.float32))
    config = {"model": {"in_dim": in_dim, "out_dim": out, "width": 16, "layers": 1, "heads": 2}, "rows": "rows.npz", "input": input_kind}
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = ContextReader(**config["model"])
    model.head.weight = mx.array(rng.standard_normal((out, 16)).astype(np.float32) * 0.1)          # a trained head is not zero
    model.save_weights(str(tmp_path / "reader.safetensors"))
    return model, np.load(tmp_path / "rows.npz"), np.load(tmp_path / "scales.npz")


@pytest.mark.parametrize("input_kind", ["reduced", "full"])
def test_rows_follow_the_trainers_formula_and_the_gain(tmp_path, input_kind):
    from drift.translate.context_reader import ContextRows
    from drift.translate.spread import restore
    model, rows, scales = _folder(tmp_path, input_kind)
    features = np.random.default_rng(5).standard_normal((7, 6)).astype(np.float32)
    reduced = (features - rows["mean"]) @ rows["basis"]
    seen = features if input_kind == "full" else reduced
    correction = np.array(model(mx.array((seen - scales["input_mean"]) / scales["component"])))
    expected = reduced @ (rows["W0"] * 1.5) + rows["b0"] + correction * scales["target"]
    reader = ContextRows(tmp_path)
    np.testing.assert_allclose(reader.read(features), expected, rtol=1e-4, atol=1e-4)
    centre, gain = np.linspace(-1, 1, 8).astype(np.float32), np.linspace(0.5, 2, 8).astype(np.float32)
    np.savez(tmp_path / "gain.npz", centre=centre, gain=gain)
    gained = ContextRows(tmp_path)
    np.testing.assert_allclose(gained.read(features), restore(expected, centre, gain), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gained.read(features, spread=False), expected, rtol=1e-4, atol=1e-4)


def test_a_gain_of_the_wrong_width_is_refused(tmp_path):
    from drift.translate.context_reader import ContextRows
    _folder(tmp_path, "reduced")
    np.savez(tmp_path / "gain.npz", centre=np.zeros(3, np.float32), gain=np.ones(3, np.float32))
    with pytest.raises(ValueError, match="output width"):
        ContextRows(tmp_path)


def test_identity_measures_whether_each_row_finds_its_own_token():
    from drift.translate.context_reader import identity, identity_loss, row_blocks
    blocks = row_blocks(2048)
    assert blocks == [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
    rng = np.random.default_rng(7)
    target = mx.array(rng.standard_normal((40, 2048)).astype(np.float32))
    shuffled = target[mx.array(np.roll(np.arange(40), 1))]
    np.testing.assert_allclose(np.array(identity(target, target, blocks)), 1.0)
    assert float(mx.max(identity(shuffled, target, blocks))) == 0.0
    rows, scale = mx.arange(40), mx.ones((2048,))
    right, wrong = identity_loss(target, target, rows, scale, 0.05, blocks), identity_loss(shuffled, target, rows, scale, 0.05, blocks)
    assert float(right) < 0.01 < float(wrong)


def test_a_long_context_is_read_in_windows_that_each_own_the_tokens_nearest_their_centre(tmp_path):
    from drift.translate.context_reader import ContextRows
    _folder(tmp_path, "full")
    features = np.random.default_rng(9).standard_normal((23, 6)).astype(np.float32)
    whole, windowed = ContextRows(tmp_path), ContextRows(tmp_path, window=8)
    np.testing.assert_allclose(ContextRows(tmp_path, window=64).read(features), whole.read(features), rtol=1e-5, atol=1e-5)   # a window longer than the context changes nothing
    rows = windowed.read(features)
    assert rows.shape == whole.read(features).shape
    np.testing.assert_allclose(rows[:7], whole.read(features[:8])[:7], rtol=1e-4, atol=1e-4)          # windows start at 0, 4, 8, 12 and 15; ties go to the earlier
    np.testing.assert_allclose(rows[7:11], whole.read(features[4:12])[3:7], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rows[-3:], whole.read(features[15:23])[-3:], rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match="two tokens"):
        ContextRows(tmp_path, window=1)
