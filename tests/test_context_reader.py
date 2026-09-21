"""Causal summaries are streaming-safe and the context reader reproduces its defining formula."""
import json
import numpy as np
import pytest
from drift.translate.context import ContextReader, causal_summaries


def test_causal_summaries_are_causal_and_chunking_does_not_change_them():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((9, 4)).astype(np.float32)
    whole, _ = causal_summaries(z, (0.5, 0.9))
    first, state = causal_summaries(z[:4], (0.5, 0.9)); second, _ = causal_summaries(z[4:], (0.5, 0.9), state)
    np.testing.assert_allclose(np.concatenate((first, second)), whole, rtol=1e-6)
    changed = z.copy(); changed[6:] += 1.0
    np.testing.assert_allclose(causal_summaries(changed, (0.5, 0.9))[0][:6], whole[:6], rtol=1e-6)      # later tokens never alter earlier summaries
    np.testing.assert_allclose(whole[0, :4], 0.5 * z[0], rtol=1e-6)


def test_context_reader_matches_its_formula_and_validates_shapes(tmp_path):
    rng = np.random.default_rng(2)
    D, r, width = 6, 2, 8
    arrays = {"W": rng.standard_normal((D + 2 * r, width)).astype(np.float32), "b": rng.standard_normal(width).astype(np.float32), "gain": np.full(width, 2.0, np.float32),
              "P": rng.standard_normal((D, r)).astype(np.float32), "g_mean": rng.standard_normal(D).astype(np.float32), "f_mean": rng.standard_normal(D + 2 * r).astype(np.float32)}
    path = tmp_path / "ctx.npz"
    np.savez(path, **arrays, meta=np.array(json.dumps({"decays": [0.5, 0.9], "summary_scale": 3.0})))
    reader = ContextReader.load(path, (3, 7), (1,), kv_heads=2, head_dim=2)
    x = rng.standard_normal((5, D)).astype(np.float32)
    k, v = reader.read({3: x[:, :3], 7: x[:, 3:]}, gain_power=1.5)[1]
    c = x - arrays["g_mean"]
    f = np.concatenate((c, causal_summaries(c @ arrays["P"], (0.5, 0.9))[0] * 3.0), 1) - arrays["f_mean"]
    np.testing.assert_allclose(np.concatenate((k.reshape(5, -1), v.reshape(5, -1)), 1), (f @ arrays["W"]) * 2.0 ** 1.5 + arrays["b"], rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="declared layers"):
        ContextReader.load(path, (3, 7), (1, 5), kv_heads=2, head_dim=2)


def test_context_fanout_adds_the_residual_before_the_gain_and_orders_rows_like_the_reader(tmp_path):
    from drift.translate.context import ContextFanoutReader
    rng = np.random.default_rng(4)
    D, r, width = 6, 2, 8
    arrays = {"W": rng.standard_normal((D + r, width)).astype(np.float32), "b": rng.standard_normal(width).astype(np.float32), "gain": np.full(width, 2.0, np.float32),
              "P": rng.standard_normal((D, r)).astype(np.float32), "g_mean": np.zeros(D, np.float32), "f_mean": np.zeros(D + r, np.float32)}
    path = tmp_path / "ctx.npz"
    np.savez(path, **arrays, meta=np.array(json.dumps({"decays": [0.5], "summary_scale": 1.0})))
    base = ContextReader.load(path, (3, 7), (1,), kv_heads=2, head_dim=2)
    count_w = np.zeros((3, 2), np.float32); count_w[0, 1] = 5.0
    fan_path = tmp_path / "fan.npz"
    R1 = rng.standard_normal((3, width)).astype(np.float32)
    np.savez(fan_path, basis=np.eye(D, dtype=np.float32)[:, :3], count_w=count_w, count_b=np.array([1.0, 0.0], np.float32), R1=R1, meta=np.array(json.dumps({"margin": 0.1, "base_sha256": base.sha256})))
    fan = ContextFanoutReader.load(fan_path, base)
    x = rng.standard_normal((4, D)).astype(np.float32) * 0.1; x[1, 0] = 3.0
    latents = {3: x[:, :3], 7: x[:, 3:]}
    out, order, which, centred = fan.read_rows(latents, 1.0)
    assert order.tolist() == [0, 1, 1, 2, 3] and which.tolist() == [0, 1, 0, 0, 0]
    f, _ = base.features(latents)
    expect = (f @ arrays["W"])[order]; expect[1] += (x @ np.eye(D, dtype=np.float32)[:, :3])[1] @ R1
    k, v = out[1]
    np.testing.assert_allclose(np.concatenate((k.reshape(5, -1), v.reshape(5, -1)), 1), expect * 2.0 + arrays["b"], rtol=1e-4, atol=1e-5)
    plain = base.read(latents, 1.0)[1]
    np.testing.assert_allclose(k[[0, 2, 3, 4]], plain[0], rtol=1e-5, atol=1e-6)                 # endpoint rows are exactly the base reader's rows
    from safetensors.numpy import save_file
    A, B, loud = rng.standard_normal((D + r, 2)).astype(np.float32), rng.standard_normal((2, width)).astype(np.float32), np.array([[0.1, -0.2]], np.float32)
    save_file({"A": A, "B": B, "loud": loud}, str(tmp_path / "corr.safetensors"))
    trained = fan.with_correction(tmp_path / "corr.safetensors")
    kt, vt = trained.read_rows(latents, 1.0)[0][1]
    raw = (f @ arrays["W"] + (f @ A) @ B)[order]; raw[1] += (x @ np.eye(D, dtype=np.float32)[:, :3])[1] @ R1
    raw = raw * 2.0; raw[:, :4] *= np.exp(0.5); raw[:, 4:] *= np.exp(-1.0)
    np.testing.assert_allclose(np.concatenate((kt.reshape(5, -1), vt.reshape(5, -1)), 1), raw + arrays["b"], rtol=1e-4, atol=1e-5)
    save_file({"A": A, "B": B[:, :7], "loud": loud}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="do not match"):
        fan.with_correction(tmp_path / "bad.safetensors")
