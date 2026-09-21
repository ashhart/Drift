"""Tokenisation-aware translation: fan-out (one writer token -> several reader tokens) and merge filtering."""
from __future__ import annotations
import json
import numpy as np
import pytest
from drift.translate.fanout import FanoutReader
from drift.translate.merge import MergeFilter
from drift.translate.stacked import StackedReader


def base_reader(tmp_path, rng):
    path = tmp_path / "base.npz"
    np.savez(path, g_mean=np.zeros(6, np.float32), W1=rng.standard_normal((6, 8)).astype(np.float32), b1=np.ones(8, np.float32), gain1=np.full(8, 2.0, np.float32))
    return StackedReader.load(path, writer_layers=(3, 7), reader_layers=(1,), kv_heads=2, head_dim=2)


def test_fanout_emits_extra_reader_entries_in_reader_order(tmp_path):
    rng = np.random.default_rng(0)
    base = base_reader(tmp_path, rng)
    basis = np.eye(6, dtype=np.float32)[:, :3]
    count_w = np.zeros((3, 3), np.float32); count_w[0, 1] = 5.0; count_w[1, 2] = 5.0      # feature 0 large -> span 2, feature 1 large -> span 3
    residual = {1: rng.standard_normal((3, 8)).astype(np.float32), 2: rng.standard_normal((3, 8)).astype(np.float32)}
    path = tmp_path / "fan.npz"
    np.savez(path, basis=basis, count_w=count_w, count_b=np.array([1.0, 0.0, 0.0], np.float32), R1=residual[1], R2=residual[2],
             meta=np.array(json.dumps({"margin": 0.1, "base_sha256": base.sha256})))
    fan = FanoutReader.load(path, base)
    x = np.zeros((3, 6), np.float32); x[1, 0] = 1.0; x[2, 1] = 1.0                        # spans 1, 2, 3
    latents = {3: x[:, :3], 7: x[:, 3:]}
    assert fan.spans(x).tolist() == [1, 2, 3]
    k, v = fan.read(latents, gain_power=1.0)[1]
    assert k.shape == (6, 2, 2)                                                           # 1 + 2 + 3 reader entries
    flat = np.concatenate((k.reshape(6, -1), v.reshape(6, -1)), 1)
    plain = lambda t: x[t] @ base.weights[1]
    expect = [plain(0), plain(1) + x[1, :3] @ residual[1], plain(1), plain(2) + x[2, :3] @ residual[2], plain(2) + x[2, :3] @ residual[1], plain(2)]
    np.testing.assert_allclose(flat, np.stack(expect) * 2.0 + 1.0, rtol=1e-5, atol=1e-5)
    other = tmp_path / "other.npz"
    np.savez(other, basis=basis, count_w=count_w, count_b=np.zeros(3, np.float32), meta=np.array(json.dumps({"margin": 0.1, "base_sha256": "0" * 64})))
    with pytest.raises(ValueError, match="different base translator"):
        FanoutReader.load(other, base)


def test_merge_filter_uses_the_next_token_and_always_keeps_the_last(tmp_path):
    path = tmp_path / "merge.npz"
    # feature 0 of the NEXT token says "I continue the same reader token" -> the current one is not an endpoint
    np.savez(path, mean=np.zeros(4, np.float32), basis=np.eye(4, dtype=np.float32)[:, :2], w=np.array([0, 0, -2.0, 0], np.float32), b=np.array(1.0),
             meta=np.array(json.dumps({"threshold": 0.5})))
    merge = MergeFilter.load(path)
    rows = np.zeros((4, 4), np.float32); rows[2, 0] = 1.0; rows[3, 0] = 1.0              # tokens 2 and 3 continue their predecessors
    assert merge.keep(rows).tolist() == [True, False, False, True]


def test_corrected_reader_reduces_to_fanout_and_applies_correction_and_loudness(tmp_path):
    from safetensors.numpy import save_file
    from drift.translate.fanout import CorrectedFanoutReader
    rng = np.random.default_rng(3)
    base = base_reader(tmp_path, rng)
    path = tmp_path / "fan.npz"
    count_w = np.zeros((3, 2), np.float32); count_w[0, 1] = 5.0
    np.savez(path, basis=np.eye(6, dtype=np.float32)[:, :3], count_w=count_w, count_b=np.array([1.0, 0.0], np.float32), R1=rng.standard_normal((3, 8)).astype(np.float32),
             meta=np.array(json.dumps({"margin": 0.1, "base_sha256": base.sha256})))
    fan = FanoutReader.load(path, base)
    x = rng.standard_normal((4, 6)).astype(np.float32); x[1, 0] = 3.0
    latents = {3: x[:, :3], 7: x[:, 3:]}
    zero = tmp_path / "zero.safetensors"
    save_file({"A": rng.standard_normal((6, 2)).astype(np.float32), "B": np.zeros((2, 8), np.float32), "loud": np.zeros((1, 2), np.float32)}, str(zero))
    plain, same = fan.read(latents, 1.5), CorrectedFanoutReader.load(zero, fan).read(latents, 1.5)
    np.testing.assert_allclose(same[1][0], plain[1][0], rtol=1e-5, atol=1e-6); np.testing.assert_allclose(same[1][1], plain[1][1], rtol=1e-5, atol=1e-6)
    A, B = rng.standard_normal((6, 2)).astype(np.float32), rng.standard_normal((2, 8)).astype(np.float32)
    trained = tmp_path / "trained.safetensors"
    save_file({"A": A, "B": B, "loud": np.array([[0.1, -0.2]], np.float32)}, str(trained))
    k, v = CorrectedFanoutReader.load(trained, fan).read(latents, 1.0)[1]
    counts = fan.spans(x).tolist()
    assert counts[1] == 2 and max(counts) == 2                                   # token 1 fans out into two reader entries
    order = [t for t, n in enumerate(counts) for _ in range(n)]
    first_of_pair = [i for i, t in enumerate(order) if i + 1 < len(order) and order[i + 1] == t]
    assert k.shape[0] == len(order)
    raw = x[order] @ base.weights[1] + ((x @ A) @ B)[order]
    for i in first_of_pair:                                                      # the j = 1 entry comes first and carries the frozen fan-out residual
        raw[i] += (x @ fan.basis)[order[i]] @ fan.residual[1]
    expect = raw * 2.0
    expect[:, :4] *= np.exp(0.5); expect[:, 4:] *= np.exp(-1.0)
    np.testing.assert_allclose(np.concatenate((k.reshape(len(order), -1), v.reshape(len(order), -1)), 1), expect + 1.0, rtol=1e-4, atol=1e-5)
    tag = rng.standard_normal((1, 2, 4)).astype(np.float32)                      # provenance training's source marker: a constant offset per layer, K and V
    tagged = tmp_path / "tagged.safetensors"
    save_file({"A": A, "B": B, "loud": np.array([[0.1, -0.2]], np.float32), "tag": tag}, str(tagged))
    kt, vt = CorrectedFanoutReader.load(tagged, fan).read(latents, 1.0)[1]
    np.testing.assert_allclose(np.concatenate((kt.reshape(len(order), -1), vt.reshape(len(order), -1)), 1), expect + 1.0 + tag.reshape(1, -1), rtol=1e-4, atol=1e-5)
    wrong_tag = tmp_path / "wrong_tag.safetensors"
    save_file({"A": A, "B": B, "loud": np.zeros((1, 2), np.float32), "tag": np.zeros((1, 2, 5), np.float32)}, str(wrong_tag))
    with pytest.raises(ValueError, match="source tag"):
        CorrectedFanoutReader.load(wrong_tag, fan)
    bad = tmp_path / "bad.safetensors"
    save_file({"A": A, "B": np.zeros((2, 9), np.float32), "loud": np.zeros((1, 2), np.float32)}, str(bad))
    with pytest.raises(ValueError, match="do not match"):
        CorrectedFanoutReader.load(bad, fan)
