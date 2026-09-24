"""The state translator fitter recovers a known linear map through its aligned-endpoint pairing."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/live/fit_state_translator.py"


def passages(folder_q, folder_g, seed, count, weight):
    rng = np.random.default_rng(seed)
    folder_q.mkdir(parents=True); folder_g.mkdir(parents=True)
    for n in range(count):
        tokens = 30
        ends_q = np.cumsum(rng.integers(1, 4, tokens))
        keep = np.sort(rng.choice(tokens, 20, replace=False))                 # GLM ends at a subset of Qwen's ends
        x = rng.standard_normal((tokens, 24)).astype(np.float16)
        y = {l: (x[keep].astype(np.float64) @ weight[l] + 0.5).astype(np.float16) for l in (0, 1)}
        np.savez(folder_q / f"p{n}.npz", offsets=np.stack([ends_q - 1, ends_q], 1).astype(np.int32), x=x)
        ends_g = ends_q[keep]
        np.savez(folder_g / f"p{n}.npz", offsets=np.stack([ends_g - 1, ends_g], 1).astype(np.int32),
                 **{f"h{l}": y[l] for l in (0, 1)})


def test_a_known_map_is_recovered_with_near_perfect_validation_fit(tmp_path):
    rng = np.random.default_rng(0)
    weight = {l: rng.standard_normal((24, 6)) for l in (0, 1)}
    passages(tmp_path / "q", tmp_path / "g", 1, 40, weight)
    passages(tmp_path / "vq", tmp_path / "vg", 2, 10, weight)
    out = tmp_path / "t.npz"
    subprocess.run([sys.executable, str(SCRIPT), "--source", str(tmp_path / "q"), "--target", str(tmp_path / "g"), "--val-source", str(tmp_path / "vq"),
                    "--val-target", str(tmp_path / "vg"), "--rank", "24", "--ridge", "1e-6", "--out", str(out)], check=True, capture_output=True)
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["train_rows"] == 800 and report["val_rows"] == 200
    assert report["r2_mean"] > 0.999
    z = np.load(out)
    x = rng.standard_normal((5, 24))
    predicted = ((x - z["mean"]) @ z["basis"].astype(np.float64)) @ z["W0"].astype(np.float64) + z["b0"]
    np.testing.assert_allclose(predicted, x @ weight[0] + 0.5, atol=0.05)


def test_per_layer_sender_arrays_are_joined_in_layer_order(tmp_path):
    rng = np.random.default_rng(3)
    weight = {l: rng.standard_normal((24, 6)) for l in (0, 1)}
    passages(tmp_path / "q", tmp_path / "g", 1, 40, weight)
    passages(tmp_path / "vq", tmp_path / "vg", 2, 10, weight)
    for folder in ("q", "vq"):                                               # split x into two sender layers, stored out of order
        for path in (tmp_path / folder).glob("*.npz"):
            z = dict(np.load(path))
            x = z.pop("x")
            np.savez(path, l11=x[:, 12:], l3=x[:, :12], **z)
    out = tmp_path / "t.npz"
    subprocess.run([sys.executable, str(SCRIPT), "--source", str(tmp_path / "q"), "--target", str(tmp_path / "g"), "--val-source", str(tmp_path / "vq"),
                    "--val-target", str(tmp_path / "vg"), "--source-key", "l", "--rank", "24", "--ridge", "1e-6", "--out", str(out)], check=True, capture_output=True)
    assert json.loads(out.with_suffix(".json").read_text())["r2_mean"] > 0.999


def grouped(folder_s, folder_r, seed, count, weight, flat=False):
    """Four sender tokens share one feature row, like DeepSeek's compressed entries; each position has its own map."""
    rng = np.random.default_rng(seed)
    folder_s.mkdir(parents=True); folder_r.mkdir(parents=True)
    for n in range(count):
        tokens = 32
        ends = np.cumsum(rng.integers(1, 4, tokens))
        x = np.repeat(rng.standard_normal((tokens // 4, 16)), 4, axis=0)
        y = np.stack([x[i] @ weight[i % 4] for i in range(tokens)]) + 0.5
        offsets = np.stack([ends - 1, ends], 1).astype(np.int32)
        np.savez(folder_s / f"p{n}.npz", offsets=offsets, x=x.astype(np.float32))
        np.savez(folder_r / f"p{n}.npz", offsets=offsets, **({"x": y.reshape(tokens, 2, 3)} if flat else {"h5": y}))


def fit(tmp_path, *extra):
    out = tmp_path / f"t{len(list(tmp_path.glob('t*.npz')))}.npz"
    subprocess.run([sys.executable, str(SCRIPT), "--source", str(tmp_path / "s"), "--target", str(tmp_path / "r"), "--val-source", str(tmp_path / "vs"),
                    "--val-target", str(tmp_path / "vr"), "--rank", "16", "--ridge", "1e-6", "--out", str(out), *extra], check=True, capture_output=True)
    return out, json.loads(out.with_suffix(".json").read_text())


def test_phases_separate_the_tokens_that_share_one_entry(tmp_path):
    from drift.translate.ridge_map import apply, load
    rng = np.random.default_rng(5)
    weight = [rng.standard_normal((16, 6)) for _ in range(4)]
    grouped(tmp_path / "s", tmp_path / "r", 1, 30, weight)
    grouped(tmp_path / "vs", tmp_path / "vr", 2, 8, weight)
    _, shared = fit(tmp_path)
    out, phased = fit(tmp_path, "--phases", "4")
    assert shared["r2_mean"] < 0.5 < 0.999 < phased["r2_mean"] and phased["phases"] == 4
    x = np.repeat(rng.standard_normal((3, 16)), 4, axis=0)
    np.testing.assert_allclose(apply(load(out), x)[5], np.stack([x[i] @ weight[i % 4] for i in range(12)]) + 0.5, atol=0.05)


def test_a_flat_receiver_array_is_one_output(tmp_path):
    from drift.translate.ridge_map import apply, load
    rng = np.random.default_rng(6)
    weight = [rng.standard_normal((16, 6))] * 4
    grouped(tmp_path / "s", tmp_path / "r", 1, 30, weight, flat=True)
    grouped(tmp_path / "vs", tmp_path / "vr", 2, 8, weight, flat=True)
    out, report = fit(tmp_path, "--target-prefix", "x")
    assert list(report["r2"]) == ["0"] and report["r2"]["0"] > 0.999
    x = rng.standard_normal((4, 16))
    np.testing.assert_allclose(apply(load(out), x)[0], x @ weight[0] + 0.5, atol=0.05)
