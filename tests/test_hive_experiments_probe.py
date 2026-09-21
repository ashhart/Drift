"""Hive experiments and probe training on dense toys."""
from __future__ import annotations
import pytest
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.types import KV
from drift.eval.hive_experiments import PoolUnit, run_departure, run_pool_handoff, run_swap_in, write_pool
from drift.mailbox.protocol import ProbeHead
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.train.probe import ProbeExample, train_probe
from drift.translate.pool import Layout, PoolFormat, Translator, fit_member, fit_pool_format


def members():
    torch.manual_seed(0)
    a = DenseAdapter(FrozenDecoder(ToyModel(seed=1, kvheads=2, dim=6, layers=3)))
    b = DenseAdapter(FrozenDecoder(ToyModel(seed=2, kvheads=1, dim=4, layers=3)))
    c = DenseAdapter(FrozenDecoder(ToyModel(seed=3, kvheads=2, dim=4, layers=3)))
    return a, b, c


def founded(a, b):
    g = torch.Generator().manual_seed(4)
    calib = torch.randint(1, 40, (60,), generator=g)
    with torch.no_grad():
        ca, cb = a.forward(calib).canonical, b.forward(calib).canonical
    la, lb = Layout("kv_split", 2, 6), Layout("kv_split", 1, 4)
    rows_a = {l: la.flatten(ca[l]) for l in range(3)}
    rows_b = {l: lb.flatten(cb[l]) for l in range(3)}
    fmt, pool_rows = fit_pool_format({l: {"a": rows_a[l], "b": rows_b[l]} for l in range(3)}, width=8)
    ta = Translator("a", la, {0: 0, 1: 1, 2: 2}, fmt)
    tb = Translator("b", lb, {0: 0, 1: 1, 2: 2}, fmt)
    fit_member(ta, rows_a, pool_rows)
    fit_member(tb, rows_b, pool_rows)
    return fmt, ta, tb, calib


def test_pool_handoff_arms_including_direct_baseline():
    a, b, _ = members()
    fmt, ta, tb, _ = founded(a, b)
    # Direct pairwise baseline: pool format == writer's native flattened layout.
    native = PoolFormat("pool.v1", 3, Layout("kv_split", 2, 6).width, "ff" * 32)
    direct = Translator("a", Layout("kv_split", 2, 6), {0: 0, 1: 1, 2: 2}, native)
    with torch.no_grad():
        for level in ("0", "1", "2"):
            direct.writers[level].linear.weight.copy_(torch.eye(24)); direct.writers[level].linear.bias.zero_()
    # The direct reader must map writer rows into the READER's layout: use tb's reader shape via a fresh reader.
    direct_reader = Translator("b", Layout("kv_split", 1, 4), {0: 0, 1: 1, 2: 2}, native)
    with torch.no_grad():
        for level in ("0", "1", "2"):
            direct_reader.readers[level].linear.weight.copy_(torch.randn(8, 24) * 0.2)
    class Direct:
        pool = native
        def write(self, canonical): return direct.write(canonical)
        def read(self, rows): return direct_reader.read(rows)
    g = torch.Generator().manual_seed(5)
    units = [PoolUnit(f"u{i}", torch.randint(1, 40, (6,), generator=g), torch.randint(1, 40, (6,), generator=g),
                      torch.randint(1, 40, (2,), generator=g)) for i in range(2)]
    rows = run_pool_handoff(a, b, ta, tb, units, new_tokens=3, direct=Direct())
    for r in rows:
        assert set(k[:-4] for k in r if k.endswith("_ids")) == {"floor", "ceiling", "pool", "hard_off", "direct"}
        assert r["hard_off_ids"] == r["floor_ids"] and r["pool_rows"] == 6


def test_swap_in_requires_enrollment_and_departure_reads_without_the_writer():
    a, b, c = members()
    fmt, ta, tb, calib = founded(a, b)
    # Enroll C against the frozen format using rows A already wrote.
    with torch.no_grad():
        pool_rows = write_pool(a, ta, calib)
        cc = c.forward(calib).canonical
    lc = Layout("kv_split", 2, 4)
    tc = Translator("c", lc, {0: 0, 1: 1, 2: 2}, fmt)
    fit_member(tc, {l: lc.flatten(cc[l]) for l in range(3)}, pool_rows)
    g = torch.Generator().manual_seed(6)
    units = [PoolUnit("s0", torch.randint(1, 40, (5,), generator=g), torch.randint(1, 40, (5,), generator=g), torch.randint(1, 40, (2,), generator=g))]
    rows = run_swap_in(a, ta, c, tc, units, new_tokens=3)
    assert rows[0]["hard_off_ids"] == rows[0]["floor_ids"] and len(rows[0]["pool_ids"]) == 3
    stranger = Translator("c", lc, {0: 0, 1: 1, 2: 2}, PoolFormat("pool.v1", 3, 8, "00" * 32))
    with pytest.raises(ValueError, match="not enrolled"):
        run_swap_in(a, ta, c, stranger, units)
    # Departure: A is gone; B reads what A left in the pool.
    left_behind = write_pool(a, ta, units[0].writer_context)
    del a
    out = run_departure(left_behind, 5, b, tb, units[0].question, new_tokens=3)
    assert out["hard_off_ids"] == out["floor_ids"] and len(out["pool_ids"]) == 3


def test_probe_reports_heldout_accuracy_and_calibration():
    torch.manual_seed(1)
    def example(label):
        base = torch.full((3, 2, 4), float(label))
        return ProbeExample(KV(base + 0.3 * torch.randn(3, 2, 4), -base + 0.3 * torch.randn(3, 2, 4)), label)
    train = [example(i % 3) for i in range(60)]
    heldout = [example(i % 3) for i in range(30)]
    probe = ProbeHead(kv_heads=2, head_dim=4, labels=3)
    report = train_probe(probe, train, heldout, steps=150)
    assert report["heldout_accuracy"] > 0.8 and 0 <= report["expected_calibration_error"] <= 1   # chance is 0.33
    assert report["audience"] == "experimenter_only" and report["chance"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        train_probe(probe, train, [ProbeExample(train[0].entry, 7)], steps=1)
