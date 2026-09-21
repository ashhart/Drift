"""M2 in-process: workers, N-member one-epoch-lag hive, checkpoints, detectors.

Members are dense toy adapters with different shapes and their own translators into
one pool format; the fitted maps are random, which is enough for mechanics.
"""
from __future__ import annotations
from uuid import UUID
import pytest
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.attention import Gate
from drift.core.pool import PoolBank
from drift.eval.detectors import DriftTracker, gate_saturation, repetition_rate
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.hive import HiveController
from drift.runtime.toy import ToyModel
from drift.runtime.worker import Worker
from drift.translate.pool import Layout, PoolFormat, Translator
from drift.transport.wire2 import FrameCodec2

SESSION = UUID(int=7)
FMT = PoolFormat("pool.v1", 3, 8, "ee" * 32)
SHAPES = {"A": dict(seed=1, kvheads=2, dim=6, layers=3), "B": dict(seed=2, kvheads=1, dim=4, layers=3),
          "C": dict(seed=3, kvheads=2, dim=4, layers=4)}


def member(name: str, writer: int, others: set[int], override=0.5, seed=0):
    torch.manual_seed(seed + writer)
    adapter = DenseAdapter(FrozenDecoder(ToyModel(**SHAPES[name])))
    d = adapter.descriptor
    level_map = {layer: layer % FMT.levels for layer in d.kv_layers[:FMT.levels]}
    translator = Translator(name, Layout("kv_split", d.kv_heads, d.head_dim), level_map, FMT)
    with torch.no_grad():
        for p in translator.parameters():
            p.mul_(0.3)
    bank = PoolBank(SESSION, translator, self_writer=writer, writers=others, sinks=2, recent=6)
    return Worker(name, writer, SESSION, adapter, translator, bank, override=override, seed=seed)


def hive(names=("A", "B"), **kw):
    ids = {name: i + 1 for i, name in enumerate(names)}
    workers = {name: member(name, ids[name], set(ids.values()) - {ids[name]}, **kw) for name in names}
    return HiveController(workers, FrameCodec2(b"h" * 32))


def run(controller, epochs, seed=11):
    g = torch.Generator().manual_seed(seed)
    outputs = []
    for _ in range(epochs):
        inputs = {name: torch.randint(1, 30, (int(torch.randint(1, 4, (1,), generator=g)),), generator=g)
                  for name in controller.workers}
        reports = controller.tick(inputs)
        outputs.append({name: r.output.clone() for name, r in reports.items()})
    return outputs


def test_two_members_run_lockstep_deterministically_and_exclude_self():
    first, second = run(hive(), 6), run(hive(), 6)
    for a, b in zip(first, second):
        for name in a:
            assert torch.equal(a[name], b[name])
    controller = hive()
    run(controller, 4)
    for name, worker in controller.workers.items():
        view = worker.bank.pin()
        assert view.epoch == 3 and worker.writer not in set(view.writers.tolist())
        assert worker.counters.epoch == 3 and worker.counters.sequence == 4
    # Foreign memory is live: epoch>0 outputs differ from an uncoupled member.
    coupled = hive()
    solo = member("A", 1, {2}, override=0.0)
    g = torch.Generator().manual_seed(11)
    for epoch in range(3):
        inputs = {name: torch.randint(1, 30, (int(torch.randint(1, 4, (1,), generator=g)),), generator=g)
                  for name in coupled.workers}
        reports = coupled.tick(inputs)
        solo_report = solo.step(inputs["A"], epoch)
        if epoch == 0:
            assert torch.equal(reports["A"].output, solo_report.output)
        else:
            assert reports["A"].foreign_tokens > 0
            assert not torch.allclose(reports["A"].output, solo_report.output)


def test_three_members_each_hold_the_other_two():
    controller = hive(("A", "B", "C"))
    run(controller, 3)
    for worker in controller.workers.values():
        writers = set(worker.bank.pin().writers.tolist())
        assert writers == {1, 2, 3} - {worker.writer}


def test_checkpoint_restore_reproduces_the_run():
    controller = hive()
    run(controller, 3, seed=5)
    checkpoint = controller.checkpoint()
    later = run(controller, 3, seed=6)
    controller.restore(checkpoint)
    again = run(controller, 3, seed=6)
    for a, b in zip(later, again):
        for name in a:
            assert torch.equal(a[name], b[name])


def test_failures_poison_and_stale_epochs_are_refused():
    controller = hive()
    run(controller, 2)
    with pytest.raises(ValueError):
        controller.tick({"A": torch.tensor([1])})           # missing member input
    assert not controller.failed                             # rejected before any state changed
    broken = controller.workers["B"]
    broken.adapter = None                                    # a worker fault mid-epoch
    with pytest.raises(AttributeError):
        run(controller, 1)
    assert controller.failed and broken.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        controller.tick({"A": torch.tensor([1]), "B": torch.tensor([2])})
    worker = member("A", 1, {2})
    worker.step(torch.tensor([3, 4]), 0)
    with pytest.raises(ValueError, match="exactly one"):
        worker.step(torch.tensor([5]), 2)
    assert not worker.poisoned                               # rejected before any state changed
    with pytest.raises(ValueError):
        Worker("A", 1, SESSION, worker.adapter, worker.translator,
               PoolBank(SESSION, worker.translator, self_writer=2, writers={1}))


def test_transport_fault_poisons_without_partial_state():
    calls = {"n": 0}

    def flaky(frame: bytes) -> bytes:
        calls["n"] += 1
        return frame if calls["n"] < 4 else frame[:-1] + bytes([frame[-1] ^ 1])

    ids = {"A": 1, "B": 2}
    workers = {n: member(n, ids[n], set(ids.values()) - {ids[n]}) for n in ids}
    controller = HiveController(workers, FrameCodec2(b"h" * 32), deliver=flaky)
    run(controller, 1)
    with pytest.raises(ValueError, match="authentication"):
        run(controller, 1)
    assert controller.failed


def test_detectors_report_numbers():
    assert repetition_rate([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], n=4) > 0.5
    assert repetition_rate(list(range(20)), n=4) == 0.0
    gates = {0: Gate(-20.0), 1: Gate(0.0), 2: Gate(20.0)}
    assert gate_saturation(gates) == {0: "closed", 1: "open", 2: "saturated"}
    tracker = DriftTracker()
    controller = hive(override=0.9)
    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        reports = controller.tick({n: torch.randint(1, 30, (2,), generator=g) for n in controller.workers})
        tracker.observe(reports["A"].canonical_norms, reports["A"].foreign_mass)
    report = tracker.report()
    assert set(report) == {"norm_drift", "mass_mean", "mass_oscillation", "nonfinite"}
    assert not report["nonfinite"] and all(0 <= m <= 1 for m in report["mass_mean"].values())
