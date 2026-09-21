"""Model packs: writer into pool.v2, reader out of it, fingerprint gating."""
from __future__ import annotations
import json
import numpy as np
import pytest
from drift.translate.pack import ModelPack


def make(tmp_path, name, layout, layers, width, pool=6, fingerprint="f" * 64, seed=0):
    rng = np.random.default_rng(seed)
    own = len(layers) * width
    path = tmp_path / f"{name}.pack.npz"
    np.savez(path, writer=rng.standard_normal((own, pool)).astype(np.float32), reader=rng.standard_normal((pool, own)).astype(np.float32),
             gain=np.full(own, 2.0, np.float32), own_mean=np.full(own, 0.5, np.float32),
             meta=np.array(json.dumps({"member": name, "layers": list(layers), "width": width, "layout": layout, "pool": {"width": pool, "fingerprint": fingerprint}})))
    return path


def test_two_members_with_different_cache_layouts_exchange_through_the_pool(tmp_path):
    glm = ModelPack.load(make(tmp_path, "glm", "mla_latent", (3, 7), 10))
    qwen = ModelPack.load(make(tmp_path, "qwen", "kv_split", (1, 5, 9), 8, seed=1), kv_heads=2)
    rng = np.random.default_rng(2)
    latents = {3: rng.standard_normal((4, 10)).astype(np.float32), 7: rng.standard_normal((4, 10)).astype(np.float32)}
    rows = glm.write(latents)
    assert rows.shape == (4, 6)
    np.testing.assert_allclose(rows, (np.concatenate((latents[3], latents[7]), 1) - 0.5) @ glm.writer, rtol=1e-5)
    entries = qwen.read(rows, glm.pool_fingerprint, gain_power=1.0)
    assert set(entries) == {1, 5, 9} and entries[5][0].shape == (4, 2, 2) and entries[5][1].shape == (4, 2, 2)
    flat = np.concatenate([np.concatenate((entries[l][0].reshape(4, -1), entries[l][1].reshape(4, -1)), 1) for l in (1, 5, 9)], 1)
    np.testing.assert_allclose(flat, (rows @ qwen.reader) * 2.0 + 0.5, rtol=1e-5)
    np.testing.assert_allclose(np.concatenate([np.concatenate((e[0].reshape(4, -1), e[1].reshape(4, -1)), 1) for e in qwen.read(rows, glm.pool_fingerprint, 0.0).values()], 1), rows @ qwen.reader + 0.5, rtol=1e-5)
    back = glm.read(qwen.write(entries), qwen.pool_fingerprint)                                   # the other direction uses the same two packs
    assert set(back) == {3, 7} and back[3].shape == (4, 10)


def test_packs_fail_closed(tmp_path):
    glm = ModelPack.load(make(tmp_path, "glm", "mla_latent", (3, 7), 10))
    other = ModelPack.load(make(tmp_path, "other", "mla_latent", (3, 7), 10, fingerprint="0" * 64))
    rows = glm.write({3: np.zeros((2, 10), np.float32), 7: np.zeros((2, 10), np.float32)})
    with pytest.raises(ValueError, match="different pool formats"):
        other.read(rows, glm.pool_fingerprint)
    with pytest.raises(ValueError, match="missing layers"):
        glm.write({3: np.zeros((2, 10), np.float32)})
    with pytest.raises(ValueError, match="same tokens"):
        glm.write({3: np.zeros((2, 10), np.float32), 7: np.zeros((3, 10), np.float32)})
    with pytest.raises(ValueError, match="nonfinite"):
        glm.write({3: np.full((2, 10), np.nan, np.float32), 7: np.zeros((2, 10), np.float32)})
    with pytest.raises(ValueError, match="pool rows"):
        glm.read(np.zeros((2, 5), np.float32), glm.pool_fingerprint)
    with pytest.raises(ValueError, match="KV head count"):
        ModelPack.load(make(tmp_path, "q", "kv_split", (1,), 8))
