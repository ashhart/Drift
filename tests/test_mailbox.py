"""M3 mechanics on dense toy members: private mail writer, controller addressing,
visibility/attention tracking, counterfactual replay. No semantic claim."""
from __future__ import annotations
from uuid import UUID
import pytest
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.pool import PoolBank
from drift.eval.replay import ReplayCondition, replay
from drift.mailbox.stream import MailStatus, MailWriter, MailboxController, merge_entries
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.hive import HiveController
from drift.runtime.toy import ToyModel
from drift.runtime.worker import Worker
from drift.translate.pool import Layout, PoolFormat, Translator
from drift.transport.wire2 import FrameCodec2

SESSION = UUID(int=8)
FMT = PoolFormat("pool.v1", 3, 8, "dd" * 32)
SHAPES = {"A": dict(seed=1, kvheads=2, dim=6, layers=3), "B": dict(seed=2, kvheads=1, dim=4, layers=3)}
NATIVE = {"A": 1, "B": 2}
MAIL = {"A": 11, "B": 12}


def member(name: str, override=0.7):
    torch.manual_seed(NATIVE[name])
    adapter = DenseAdapter(FrozenDecoder(ToyModel(**SHAPES[name])))
    d = adapter.descriptor
    translator = Translator(name, Layout("kv_split", d.kv_heads, d.head_dim), {l: l for l in d.kv_layers}, FMT)
    with torch.no_grad():
        for p in translator.parameters():
            p.mul_(0.3)
    others = {v for k, v in NATIVE.items() if k != name}
    mail_others = {v for k, v in MAIL.items() if k != name}
    return Worker(name, NATIVE[name], SESSION, adapter, translator,
                  PoolBank(SESSION, translator, NATIVE[name], others, sinks=2, recent=6),
                  override=override, mail_writer=MAIL[name],
                  mail_bank=PoolBank(SESSION, translator, MAIL[name], mail_others, sinks=0, recent=8, multiple_per_epoch=True),
                  mail=MailWriter(adapter.decoder.model.config.hidden_size, max_tokens=4))


def hive(**kw):
    return HiveController({n: member(n, **kw) for n in ("A", "B")}, FrameCodec2(b"m" * 32))


def inputs(seed, n=2):
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randint(1, 30, (n,), generator=g) for name in ("A", "B")}


def test_mail_writer_uses_a_private_branch_and_counts_tokens():
    worker = member("A")
    worker.step(torch.tensor([1, 2, 3]), 0)
    before = worker.adapter.snapshot_state(worker.state)
    canonical = worker.mail.write(worker.adapter, worker.state, torch.tensor([4, 5]))
    after = worker.adapter.snapshot_state(worker.state)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(entry.tokens == 3 for entry in canonical.values())         # marker + 2 local tokens
    assert worker.mail.local_tokens_consumed == 2 and worker.mail.forward_passes == 1
    with pytest.raises(ValueError):
        worker.mail.write(worker.adapter, worker.state, torch.arange(1, 9))
    loss = sum(e.v.square().sum() for e in canonical.values())
    loss.backward()
    assert worker.mail.marker.grad is not None and worker.mail.marker.grad.abs().sum() > 0


def test_controller_assigns_ids_slots_caps_ttl_and_metrics():
    box = MailboxController(ttl=3, slots=2, per_epoch_cap=2)
    a = box.create("A", 0, tokens=3)
    b = box.create("A", 0, tokens=2)
    assert (a.id, a.slot, b.id, b.slot) == (0, 0, 1, 1)
    with pytest.raises(OverflowError, match="cap"):
        box.create("A", 0, tokens=1)
    with pytest.raises(OverflowError, match="slot"):
        box.create("B", 1, tokens=1)
    box.cancel(b.id)
    c = box.create("B", 1, tokens=1)
    assert c.slot == 1
    box.expire(3)
    assert box.records[a.id].status == MailStatus.EXPIRED and box.records[c.id].status == MailStatus.PENDING
    with pytest.raises(ValueError, match="late masking"):
        box.record_causal_use(c.id, 2, active_correct=True, ablated_correct=False, replay_started_before_exposure=False)
    box.record_causal_use(c.id, 2, active_correct=True, ablated_correct=False, replay_started_before_exposure=True)
    m = box.metrics()
    assert m["counts"]["causally_incorporated"] == 1 and m["counts"]["expired_without_incorporation"] == 1
    assert m["censored_at_ttl"] == [3] and m["incorporation_latency_epochs"] == [1]


def test_mail_flows_between_members_and_is_tracked():
    controller, box = hive(), MailboxController(ttl=5, slots=4, per_epoch_cap=2)
    controller.tick(inputs(1))
    controller.tick(inputs(2))
    record = box.create("A", epoch=1, tokens=2)
    receiver = controller.workers["B"]
    controller.post_mail("A", torch.tensor([7, 8]), box, record.id)
    # registered before commit: positions are the slots the bank was about to assign
    assert receiver.mail_bank.next_slot == 3
    assert box.positions[("B", MAIL["A"])] == [(0, 0), (1, 0), (2, 0)]
    reports = controller.tick(inputs(3))
    b = reports["B"]
    assert b.mail_foreign_tokens == 3 and b.native_foreign_tokens > 0
    assert all(m.numel() == b.native_foreign_tokens + 3 for m in b.entry_mass.values())
    mail_mass = {layer: m[b.native_foreign_tokens:] for layer, m in b.entry_mass.items()}
    box.observe("B", 2, receiver.mail_bank.pin(), mail_mass)
    assert box.records[0].status == MailStatus.ATTENDED and box.records[0].visible == 2
    assert box.metrics()["attention_latency_epochs"] == [1]
    # The sender's own mail bank never holds its own mail.
    assert controller.workers["A"].mail_bank.pin() is None
    # Mail rides its own cursor, not the native one.
    assert controller.workers["A"].counters.mail_sequence == 1 and controller.workers["A"].counters.sequence == 3


def test_counterfactual_replay_forks_before_exposure():
    controller = hive()
    controller.tick(inputs(1))
    controller.tick(inputs(2))
    controller.post_mail("A", torch.tensor([7, 8]))
    checkpoint = controller.checkpoint()                    # B has not yet pinned the mail
    schedule = [inputs(3), inputs(4)]
    mail_positions = frozenset(controller.workers["B"].mail_bank.pin().positions.tolist())
    conditions = [ReplayCondition("active"), ReplayCondition("ablated", mail_positions)]
    results = replay(controller, checkpoint, "B", lambda name: controller.workers[name].mail_bank, conditions, schedule)
    assert results["active"][0].mail_foreign_tokens == 3 and results["ablated"][0].mail_foreign_tokens == 0
    assert not torch.allclose(results["active"][-1].output, results["ablated"][-1].output)
    # Replay is repeatable and leaves the controller at the fork.
    again = replay(controller, checkpoint, "B", lambda name: controller.workers[name].mail_bank, conditions, schedule)
    assert torch.equal(again["active"][-1].output, results["active"][-1].output)
    assert controller.epoch == 2 and not controller.failed


def test_merge_entries_orders_mail_after_native():
    from types import MappingProxyType
    from drift.core.pool import PoolView
    from drift.core.types import KV
    native = PoolView(0, torch.tensor([0, 3]), torch.tensor([1, 1]), MappingProxyType({0: KV(torch.ones(2, 1, 2), torch.ones(2, 1, 2))}))
    mail = PoolView(0, torch.tensor([0]), torch.tensor([11]), MappingProxyType({0: KV(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))}))
    merged = merge_entries(native, mail)
    assert merged.positions.tolist() == [0, 3, 4] and merged.layers[0].k.shape == (3, 1, 2)
    assert merge_entries(None, None) is None


def test_two_mails_from_one_sender_in_one_epoch_are_both_delivered():
    """The per-epoch cap allows more than one mail; the receivers' mail banks must accept them
    and the hive must not be poisoned."""
    controller, box = hive(), MailboxController(ttl=5, slots=4, per_epoch_cap=2)
    controller.tick(inputs(1))
    controller.tick(inputs(2))
    first = box.create("A", 1, tokens=2)
    controller.post_mail("A", torch.tensor([7, 8]), box, first.id)
    second = box.create("A", 1, tokens=2)
    controller.post_mail("A", torch.tensor([9, 10]), box, second.id)
    assert not controller.failed
    view = controller.workers["B"].mail_bank.pin()
    assert view.positions.numel() == 6 and view.writers.unique().tolist() == [MAIL["A"]]
    reports = controller.tick(inputs(3))
    assert reports["B"].mail_foreign_tokens == 6


def test_slot_release_only_by_its_current_owner():
    """An old record's cancel/expire must not free a slot that a newer message now holds."""
    box = MailboxController(ttl=1, slots=1, per_epoch_cap=3)
    old = box.create("A", 0, tokens=1)
    box.expire(1)                                   # old expired: slot 0 free
    new = box.create("B", 1, tokens=1)
    assert new.slot == 0 and box.slot_owner[0] == new.id
    box.cancel(old.id)                              # already expired; must not touch slot 0
    assert box.slot_owner.get(0) == new.id
    with pytest.raises(OverflowError, match="slot"):
        box.create("A", 1, tokens=1)                # slot 0 is still taken by `new`
    box.reject(old.id)
    box.record_causal_use(new.id, 1, active_correct=True, ablated_correct=False, replay_started_before_exposure=True)
    assert 0 not in box.slot_owner                  # released by its real owner
