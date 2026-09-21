"""Shared-pool translator mechanics on synthetic members with a known latent.

Three synthetic members observe the same hidden 6-dim signal through different
affine "architectures": A (kv_split 2x4), B (mla_latent 12), C (kv_split 1x10).
Nothing here is language; it checks fitting, cross-reading, enrollment isolation,
layouts and persistence.
"""
from __future__ import annotations
import pytest
import torch
from drift.core.types import KV
from drift.translate.pool import (Layout, PoolFormat, Translator, cross_error, fit_member,
                                      fit_pool_format, load_translator, save_translator)

LEVELS = 2
SIGNAL = 6


@pytest.fixture
def world():
    torch.manual_seed(3)
    layouts = {"A": Layout("kv_split", 2, 4), "B": Layout("mla_latent", 1, 12), "C": Layout("kv_split", 1, 10)}
    maps = {name: {lvl: (torch.randn(SIGNAL, lay.width), torch.randn(lay.width) * 0.3) for lvl in range(LEVELS)}
            for name, lay in layouts.items()}

    def observe(name: str, z: torch.Tensor) -> dict[int, torch.Tensor]:
        return {lvl: z @ maps[name][lvl][0] + maps[name][lvl][1] + 0.01 * torch.randn(z.shape[0], layouts[name].width)
                for lvl in range(LEVELS)}

    train_z, test_z = torch.randn(400, SIGNAL), torch.randn(50, SIGNAL)
    return layouts, observe, train_z, test_z


def test_founding_members_cross_read_through_the_pool(world):
    layouts, observe, train_z, test_z = world
    rows_a, rows_b = observe("A", train_z), observe("B", train_z)
    fmt, pool_rows = fit_pool_format({lvl: {"A": rows_a[lvl], "B": rows_b[lvl]} for lvl in range(LEVELS)}, width=SIGNAL)
    assert fmt.levels == LEVELS and fmt.width == SIGNAL and len(fmt.fingerprint) == 64
    a = Translator("A", layouts["A"], {1: 0, 3: 1}, fmt)
    b = Translator("B", layouts["B"], {2: 0, 6: 1}, fmt)
    ra = fit_member(a, rows_a, pool_rows)
    rb = fit_member(b, rows_b, pool_rows)
    for report in (ra, rb):
        for level in range(LEVELS):
            assert report[level]["roundtrip_rmse"] < 0.05 * report[level]["baseline_rmse"]
    held_a, held_b = observe("A", test_z), observe("B", test_z)
    for level, err in cross_error(a, b, held_a, held_b).items():
        assert err["cross_rmse"] < 0.05 * err["baseline_rmse"]
    for level, err in cross_error(b, a, held_b, held_a).items():
        assert err["cross_rmse"] < 0.05 * err["baseline_rmse"]


def test_write_and_read_respect_layouts_and_level_maps(world):
    layouts, observe, train_z, _ = world
    rows_a, rows_b = observe("A", train_z), observe("B", train_z)
    fmt, pool_rows = fit_pool_format({lvl: {"A": rows_a[lvl], "B": rows_b[lvl]} for lvl in range(LEVELS)}, width=SIGNAL)
    a = Translator("A", layouts["A"], {1: 0, 3: 1}, fmt)
    b = Translator("B", layouts["B"], {2: 0, 6: 1}, fmt)
    fit_member(a, rows_a, pool_rows)
    fit_member(b, rows_b, pool_rows)
    canonical_a = {1: layouts["A"].unflatten(rows_a[0][:7]), 3: layouts["A"].unflatten(rows_a[1][:7])}
    assert isinstance(canonical_a[1], KV) and canonical_a[1].k.shape == (7, 2, 4)
    pool = a.write(canonical_a)
    assert set(pool) == {0, 1} and pool[0].shape == (7, SIGNAL)
    read_b = b.read(pool)
    assert set(read_b) == {2, 6} and read_b[2].shape == (7, 12)
    torch.testing.assert_close(read_b[2], rows_b[0][:7], atol=0.15, rtol=0)
    with pytest.raises(ValueError, match="missing canonical"):
        a.write({1: canonical_a[1]})
    with pytest.raises(ValueError, match="absent"):
        b.read({0: pool[0]})
    with pytest.raises(ValueError):
        a.write({1: rows_a[0][:7], 3: canonical_a[3]})     # wrong layout for kv_split


def test_enrollment_changes_only_the_new_member(world):
    layouts, observe, train_z, test_z = world
    rows_a, rows_b = observe("A", train_z), observe("B", train_z)
    fmt, pool_rows = fit_pool_format({lvl: {"A": rows_a[lvl], "B": rows_b[lvl]} for lvl in range(LEVELS)}, width=SIGNAL)
    a = Translator("A", layouts["A"], {1: 0, 3: 1}, fmt)
    b = Translator("B", layouts["B"], {2: 0, 6: 1}, fmt)
    fit_member(a, rows_a, pool_rows)
    fit_member(b, rows_b, pool_rows)
    frozen = {(t.member, k): v.clone() for t in (a, b) for k, v in t.state_dict().items()}
    # Enrollment: C sees calibration text the founders already wrote into the pool.
    cal_z = torch.randn(300, SIGNAL)
    pool_from_a = {lvl: a.writers[str(lvl)](observe("A", cal_z)[lvl]).detach() for lvl in range(LEVELS)}
    c = Translator("C", layouts["C"], {0: 0, 5: 1}, fmt)
    rc = fit_member(c, observe("C", cal_z), pool_from_a)
    for level in range(LEVELS):
        assert rc[level]["roundtrip_rmse"] < 0.1 * rc[level]["baseline_rmse"]
    for t in (a, b):
        for name, tensor in t.state_dict().items():
            assert torch.equal(tensor, frozen[(t.member, name)])
    held_a, held_c, held_b = observe("A", test_z), observe("C", test_z), observe("B", test_z)
    for err in cross_error(a, c, held_a, held_c).values():
        assert err["cross_rmse"] < 0.1 * err["baseline_rmse"]
    for err in cross_error(c, b, held_c, held_b).values():
        assert err["cross_rmse"] < 0.1 * err["baseline_rmse"]


def test_persistence_binds_to_the_pool_fingerprint(world, tmp_path):
    layouts, observe, train_z, _ = world
    rows_a, rows_b = observe("A", train_z), observe("B", train_z)
    fmt, pool_rows = fit_pool_format({lvl: {"A": rows_a[lvl], "B": rows_b[lvl]} for lvl in range(LEVELS)}, width=SIGNAL)
    a = Translator("A", layouts["A"], {1: 0, 3: 1}, fmt, kind="mlp", rank=8)
    fit_member(a, rows_a, pool_rows)
    save_translator(a, tmp_path / "A", {"note": "test"})
    loaded = load_translator(tmp_path / "A", fmt)
    for name, tensor in a.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    assert loaded.level_map == {1: 0, 3: 1} and loaded.kind == "mlp"
    other = PoolFormat(fmt.version, fmt.levels, fmt.width, "0" * 64)
    with pytest.raises(ValueError, match="different pool format"):
        load_translator(tmp_path / "A", other)
    (tmp_path / "A" / "translator.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_translator(tmp_path / "A", fmt)


def test_format_fitting_rejects_misaligned_or_insufficient_rows(world):
    layouts, observe, train_z, _ = world
    rows_a, rows_b = observe("A", train_z), observe("B", train_z)
    with pytest.raises(ValueError, match="aligned rows"):
        fit_pool_format({0: {"A": rows_a[0], "B": rows_b[0][:10]}}, width=SIGNAL)
    with pytest.raises(ValueError, match="aligned rows"):
        fit_pool_format({0: {"A": rows_a[0][:4]}}, width=SIGNAL)
    with pytest.raises(ValueError):
        Translator("A", layouts["A"], {1: 0, 3: 0}, PoolFormat("pool.v1", 2, 6, "f" * 64))
    with pytest.raises(ValueError):
        Translator("A", layouts["A"], {1: 5}, PoolFormat("pool.v1", 2, 6, "f" * 64))
