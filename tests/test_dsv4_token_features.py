"""DeepSeek V4 per-token features: each token gets its group's decoded entries, its neighbours' and its indexer keys."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from drift.translate.dsv4_pages import encode_entries

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/live/dsv4_token_features.py"


def capture(folder, values, keys, tokens=10):
    """One passage: two ratio-4 layers with indexers and a ratio-128 layer, as dsv4_capture_rows.py saves them."""
    manifest, arrays = [], {}
    for layer in (2, 4):
        manifest.append({"name": f"model.layers.{layer}.attn", "ratio": 4})
        arrays[f"r{len(manifest) - 1}"] = encode_entries(values[layer], 64)
        manifest.append({"name": f"model.layers.{layer}.attn.indexer.k_cache", "ratio": 4})
        page = np.concatenate((np.full((64, 128), 0x38, np.uint8).reshape(-1), np.full(64, keys[layer], "<f4").view(np.uint8)))
        arrays[f"r{len(manifest) - 1}"] = page.reshape(64, 132)
    manifest.append({"name": "model.layers.3.attn", "ratio": 128})
    arrays[f"r{len(manifest) - 1}"] = encode_entries(np.ones((2, 512), np.float32), 2)
    offsets = np.stack([np.arange(tokens), np.arange(tokens) + 1], 1).astype(np.int32)
    np.savez(folder / "p0.npz", offsets=offsets, span=np.array([256, 256, tokens], np.int32),
             manifest=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8), **arrays)


def run(tmp_path, *extra):
    out = tmp_path / f"out{len(extra)}"
    subprocess.run([sys.executable, str(SCRIPT), "--captures", str(tmp_path / "c"), "--out", str(out), *extra],
                   check=True, capture_output=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
    return np.load(out / "p0.npz")


def test_tokens_take_their_group_entries_neighbours_and_indexer_keys(tmp_path):
    (tmp_path / "c").mkdir()
    values = {layer: np.zeros((64, 512), np.float32) for layer in (2, 4)}
    for layer in (2, 4):
        values[layer][:, :448] = (np.arange(64)[:, None] + layer * 100).astype(np.float32)   # exact in fp8 up to its step
    capture(tmp_path / "c", values, {2: 0.5, 4: 2.0})
    plain = run(tmp_path)
    assert plain["x"].shape == (10, 2 * 448) and np.array_equal(plain["offsets"][:, 1], np.arange(1, 11))
    decoded = plain["x"].astype(np.float32)
    assert np.allclose(decoded[5, :448], values[2][1, :448], rtol=0.07) and np.allclose(decoded[9, 448:], values[4][2, :448], rtol=0.07)
    assert np.array_equal(decoded[4], decoded[7]) and not np.array_equal(decoded[3], decoded[4])
    rich = run(tmp_path, "--context", "1", "--indexer")
    x = rich["x"].astype(np.float32)
    assert x.shape == (10, 3 * 2 * 448 + 2 * 128)
    assert not x[0, 896:1792].any() and np.array_equal(x[0, 1792:2688], decoded[4])        # no group before the first; the next group after
    assert np.allclose(x[0, 2688:2816], 0.5) and np.allclose(x[0, 2816:], 2.0)            # code 0x38 is 1.0 in e4m3fn
