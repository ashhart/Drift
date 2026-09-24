"""Stacked-level reader and the fp8_ds_mla handoff reader."""
from __future__ import annotations
import json
import struct
import numpy as np
import pytest
from drift.serving.glm53_delta import read_latents_span, stitch
from drift.serving.glm53_handoff import E4M3FN, MAGIC, read_header, dequantize_fp8_ds_mla, pack_fp8_ds_mla, read_latents
from drift.translate.stacked import StackedReader


def artifact(tmp_path, rng, inputs=12, width=8, shrink=0.5):
    true = rng.standard_normal((inputs, width)).astype(np.float32)
    path = tmp_path / "s.npz"
    np.savez(path, g_mean=np.full(inputs, 0.25, np.float32), meta=np.array(json.dumps({"ridge": 0.1})),
             **{f"W{l}": true * shrink for l in (1, 5)}, **{f"b{l}": np.full(width, 2.0, np.float32) for l in (1, 5)},
             **{f"gain{l}": np.full(width, 1 / shrink, np.float32) for l in (1, 5)})
    return path, true


def test_gain_restores_the_scale_a_shrunk_regression_lost(tmp_path):
    rng = np.random.default_rng(0)
    path, true = artifact(tmp_path, rng)
    reader = StackedReader.load(path, writer_layers=(3, 7), reader_layers=(1, 5), kv_heads=2, head_dim=2)
    latents = {3: rng.standard_normal((9, 6)).astype(np.float32), 7: rng.standard_normal((9, 6)).astype(np.float32)}
    target = (np.concatenate((latents[3], latents[7]), 1) - 0.25) @ true + 2.0
    shrunk, restored = reader.read(latents, gain_power=0.0), reader.read(latents, gain_power=1.0)
    k, v = restored[5]
    assert k.shape == (9, 2, 2) and v.shape == (9, 2, 2)
    np.testing.assert_allclose(np.concatenate((k.reshape(9, -1), v.reshape(9, -1)), 1), target, rtol=1e-4, atol=1e-4)
    assert np.abs(shrunk[5][0] - 2.0).mean() < 0.6 * np.abs(k - 2.0).mean()
    assert reader.meta == {"ridge": 0.1} and len(reader.sha256) == 64


def test_stacked_reader_rejects_incomplete_or_mismatched_input(tmp_path):
    rng = np.random.default_rng(1)
    path, _ = artifact(tmp_path, rng)
    reader = StackedReader.load(path, (3, 7), (1, 5), 2, 2)
    good = {3: np.zeros((4, 6), np.float32), 7: np.zeros((4, 6), np.float32)}
    with pytest.raises(ValueError, match="missing writer layers"):
        reader.read({3: good[3]})
    with pytest.raises(ValueError, match="same tokens"):
        reader.read({3: good[3], 7: np.zeros((5, 6), np.float32)})
    with pytest.raises(ValueError, match="width"):
        reader.read({3: good[3], 7: np.zeros((4, 7), np.float32)})
    with pytest.raises(ValueError, match="nonfinite"):
        reader.read({3: good[3], 7: np.full((4, 6), np.nan, np.float32)})
    with pytest.raises(ValueError, match="shapes"):
        StackedReader.load(path, (3, 7), (1, 5), 2, 4)


def test_e4m3fn_table_matches_the_format():
    assert E4M3FN[0] == 0 and E4M3FN[0x38] == 1.0 and E4M3FN[0xB8] == -1.0
    assert E4M3FN[0x7E] == 448.0 and E4M3FN[0x01] == 2.0 ** -9 and np.isnan(E4M3FN[0x7F])
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch has no float8_e4m3fn")
    reference = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    keep = ~np.isnan(reference)
    assert np.array_equal(np.isnan(reference), np.isnan(E4M3FN)) and np.array_equal(reference[keep], E4M3FN[keep])


def blob(tmp_path, n_tokens, layers, extra=()):
    rng = np.random.default_rng(2)
    tensors, payload, truth = [], b"", {}
    for name in layers:
        codes = rng.integers(0, 0x7E, size=(1, 8, 512), dtype=np.uint8)
        scales = rng.uniform(0.01, 0.1, size=(1, 8, 4)).astype("<f4")
        page = np.concatenate((codes, scales.view(np.uint8), np.zeros((1, 8, 128), np.uint8)), axis=2)
        tensors.append({"layer": name, "shape": [1, 8, 656], "dtype": "uint8", "offset": len(payload), "nbytes": page.nbytes})
        payload += page.tobytes()
        truth[name] = (E4M3FN[codes[0]].reshape(8, 4, 128) * scales[0][:, :, None]).reshape(8, 512)[:n_tokens]
    tensors += list(extra)
    header = json.dumps({"format": "glm53-handoff-raw-v1", "n_tokens": n_tokens, "export_from": 0, "tp_rank": 0, "tp_size": 2,
                         "cache_config": {"cache_dtype": "fp8_ds_mla"}, "prompt_token_ids": [11, 12, 13], "tensors": tensors}).encode()
    start = -(-(16 + len(header)) // 4096) * 4096
    path = tmp_path / "rank0.bin"
    path.write_bytes(MAGIC + struct.pack("<Q", len(header)) + header + b"\0" * (start - 16 - len(header)) + payload)
    return path, truth


def test_handoff_reader_returns_only_target_attention_latents(tmp_path):
    names = ["language_model.model.layers.3.self_attn.attn", "language_model.model.layers.7.self_attn.attn", "model.layers.45.self_attn.attn"]
    path, truth = blob(tmp_path, 5, names, extra=[{"layer": "language_model.model.layers.4.linear_attn", "kind": "state", "shape": [1], "dtype": "uint8", "offset": 0, "nbytes": 1}])
    latents = read_latents(path)
    assert sorted(latents) == [3, 7] and latents[3].shape == (5, 512) and latents[3].dtype == np.float32      # drafter layer 45 and state pages skipped
    np.testing.assert_allclose(latents[7], truth[names[1]], rtol=1e-6)
    assert not any(isinstance(v, (list, tuple)) for v in latents.values())                                      # token ids never come back


def test_handoff_reader_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="row shape"):
        dequantize_fp8_ds_mla(np.zeros((2, 100), np.uint8))
    bad = np.zeros((1, 656), np.uint8); bad[0, 0] = 0x7F; bad[0, 512:516] = np.frombuffer(np.float32(1).tobytes(), np.uint8)
    with pytest.raises(ValueError, match="nonfinite"):
        dequantize_fp8_ds_mla(bad)
    path, _ = blob(tmp_path, 9, ["language_model.model.layers.3.self_attn.attn"])
    with pytest.raises(ValueError, match="do not cover"):
        read_latents(path)
    (tmp_path / "junk.bin").write_bytes(b"nothing here")
    with pytest.raises(ValueError, match="not a glm53"):
        read_latents(tmp_path / "junk.bin")


def test_pack_is_the_inverse_of_dequantize_up_to_fp8_rounding():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((40, 512)) * rng.uniform(0.05, 3.0, size=(40, 1))).astype(np.float32)
    x[0] = 0.0
    packed = pack_fp8_ds_mla(x)
    assert packed.shape == (40, 528) and packed.dtype == np.uint8
    rows = np.concatenate((packed, np.zeros((40, 128), np.uint8)), axis=1)
    back = dequantize_fp8_ds_mla(rows)
    assert np.array_equal(back[0], np.zeros(512, np.float32))
    assert np.abs(back - x).max() <= 0.0625 * np.abs(x).max(axis=1, keepdims=True).max() + 1e-6           # e4m3 step is <= 1/16 of the value
    assert np.sqrt(((back - x) ** 2).mean()) / np.sqrt((x ** 2).mean()) < 0.03
    np.testing.assert_array_equal(pack_fp8_ds_mla(back), packed)                                          # idempotent on representable values
    torch = pytest.importorskip("torch")
    if hasattr(torch, "float8_e4m3fn"):
        scales = packed[:, 512:].copy().view("<f4")
        reference = (torch.from_numpy(x).reshape(40, 4, 128) / torch.from_numpy(scales)[:, :, None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(40, 512).numpy()
        assert (reference == packed[:, :512]).mean() > 0.999                                               # ties may round differently
    with pytest.raises(ValueError):
        pack_fp8_ds_mla(np.full((2, 512), np.inf, np.float32))


def test_stacked_reader_can_emit_mla_latents(tmp_path):
    rng = np.random.default_rng(4)
    path = tmp_path / "r.npz"
    W = rng.standard_normal((8, 6)).astype(np.float32)
    np.savez(path, g_mean=np.zeros(8, np.float32), W3=W, b3=np.ones(6, np.float32), gain3=np.ones(6, np.float32))
    reader = StackedReader.load(path, writer_layers=(1, 2), reader_layers=(3,), kv_heads=0, head_dim=6)
    x = {1: rng.standard_normal((5, 4)).astype(np.float32), 2: rng.standard_normal((5, 4)).astype(np.float32)}
    np.testing.assert_allclose(reader.read(x)[3], np.concatenate((x[1], x[2]), 1) @ W + 1.0, rtol=1e-5)


def test_a_delta_export_starts_at_its_first_block_and_stitches_onto_a_prefix_export(tmp_path):
    name = "language_model.model.layers.3.self_attn.attn"
    full, truth = blob(tmp_path, 8, [name])
    header, start = read_header(full)
    header.update(export_from=8, n_tokens=16)                                                                  # one page of 8 slots from position 8
    header["tensors"][0].update(first_block_index=1, pages_per_block=1)
    body = json.dumps(header).encode()
    delta = tmp_path / "delta.bin"
    delta.write_bytes(MAGIC + struct.pack("<Q", len(body)) + body + b"\0" * (start - 16 - len(body)) + full.read_bytes()[start:])
    with pytest.raises(ValueError, match="delta exports"):
        read_latents(delta)
    first, later = read_latents_span(delta)
    assert first == 8 and later[3].shape == (8, 512)                                                          # positions 8 to 15
    np.testing.assert_allclose(later[3], truth[name], rtol=1e-6)
    prefix = {3: np.zeros((10, 512), np.float32)}
    joined = stitch(prefix, first, later)
    assert joined[3].shape == (16, 512) and not joined[3][:8].any()
    with pytest.raises(ValueError, match="reach"):
        stitch({3: np.zeros((7, 512), np.float32)}, first, later)

