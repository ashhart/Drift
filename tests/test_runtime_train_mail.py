from types import MappingProxyType
import pytest
import torch
from drift.core.types import KV
from drift.core.attention import Gate
from drift.core.memory import ForeignView
from drift.core.projector import BridgeProjector
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.runtime.manifest import parameter_digest
from drift.train.fit import receiver_kl, fit_mlp, save_projector, load_projector
from drift.train.alignment import shared_causal_endpoints
from drift.mailbox.protocol import ThoughtWriter, MailboxLedger, MailState, ProbeHead
from drift.plugin.control import ControlEvent, Op
from drift.eval.metrics import paired_bootstrap, e2_gate, CostLedger


def test_prefill_equals_incremental_and_self_import():
    worker = FrozenDecoder(ToyModel())
    ids = torch.tensor([1, 4, 7, 2, 5, 8], dtype=torch.long)
    full, prefix = worker.forward(ids), worker.forward(ids[:4])
    native = worker.forward(ids[4:], prefix.state)
    imported = worker.import_self_prefix(prefix.canonical_delta, torch.arange(4))
    handoff = worker.forward(ids[4:], imported)
    torch.testing.assert_close(full.logits[4:], native.logits)
    torch.testing.assert_close(handoff.logits, native.logits)
    with pytest.raises(ValueError): worker.import_self_prefix({0: prefix.canonical_delta[0]}, torch.arange(4))


def test_closed_foreign_bank_does_not_alter_native_weights_or_output():
    worker = FrozenDecoder(ToyModel())
    before = parameter_digest(worker.model)
    source = worker.forward(torch.tensor([1, 2, 3]))
    memory = ForeignView(0, torch.arange(3), source.canonical_delta)
    ids = torch.tensor([5, 6])
    baseline, closed = worker.forward(ids), worker.forward(ids, foreign=memory, override=0)
    assert torch.equal(baseline.logits, closed.logits)
    assert parameter_digest(worker.model) == before


def test_gradients_pass_through_frozen_decoder_to_sidecars():
    worker = FrozenDecoder(ToyModel())
    before = parameter_digest(worker.model)
    projector, gate = BridgeProjector((1, 4), (2, 6), "mlp", rank=8), Gate(-1)
    source = KV(torch.randn(4, 1, 4), torch.randn(4, 1, 4))
    projected = projector(source)
    view = ForeignView(0, torch.arange(4), MappingProxyType({1: projected}))
    out = worker.forward(torch.tensor([1, 2, 3]), foreign=view, gates={1: gate})
    teacher = out.logits.detach().clone(); teacher[:, 5] += 4.0
    loss = receiver_kl(out.logits, teacher)
    loss.backward()
    assert gate.logit.grad is not None and gate.logit.grad.abs() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters())
    assert all(p.grad is None and not p.requires_grad for p in worker.model.parameters())
    assert before == parameter_digest(worker.model)


def test_receiver_kl_rejects_detached_or_different_vocab():
    with pytest.raises(ValueError): receiver_kl(torch.zeros(2, 7), torch.zeros(2, 7))
    with pytest.raises(ValueError): receiver_kl(torch.zeros(2, 7, requires_grad=True), torch.zeros(2, 8))


def test_private_mailbox_does_not_modify_parent_and_marker_trains():
    worker, writer = FrozenDecoder(ToyModel()), ThoughtWriter(24)
    state = worker.forward(torch.tensor([1, 2, 3])).state
    old = state.layers[0].k.clone()
    kv = writer.write(worker, state, torch.tensor([4, 5]))
    assert state.length == 3 and torch.equal(old, state.layers[0].k)
    assert all(x.tokens == 3 for x in kv.values())  # marker + two local tokens
    sum(x.v.square().sum() for x in kv.values()).backward()
    assert writer.marker.grad is not None and writer.marker.grad.abs().sum() > 0


def test_mail_attended_is_not_delivered_and_starvation_is_counted():
    ledger = MailboxLedger(capacity=1); ledger.create(3, 0, 5)
    ledger.observe_attention(3, 1, mass=0.9)
    assert ledger.records[3].state == MailState.ATTENDED
    with pytest.raises(OverflowError): ledger.create(4, 1, 5)
    with pytest.raises(ValueError):
        ledger.record_causal_use(3, 2, active_correct=True, ablated_correct=False,
                                 replay_started_before_exposure=False)
    ledger.expire(5)
    assert ledger.records[3].state == MailState.STARVED
    ledger.create(4, 5, 5)
    ledger.record_causal_use(4, 6, active_correct=True, ablated_correct=False,
                             replay_started_before_exposure=True)
    assert ledger.records[4].incorporation == 6


def test_probe_detaches_features_from_models():
    kv = KV(torch.randn(3, 2, 4, requires_grad=True), torch.randn(3, 2, 4, requires_grad=True))
    probe = ProbeHead(2, 4, 7); probe(kv).sum().backward()
    assert kv.k.grad is None and kv.v.grad is None and probe.head.weight.grad is not None


def test_unicode_causal_endpoint_alignment():
    pairs = shared_causal_endpoints("aé🙂z", [(0, 1), (1, 3), (3, 4)], [(0, 2), (2, 3), (3, 4)])
    assert pairs == [(1, 1, 7), (2, 2, 8)]
    assert shared_causal_endpoints("abc", [(0, 1)], [(0, 2)]) == []
    with pytest.raises(ValueError): shared_causal_endpoints("a", [(0, 2)], [(0, 1)])


def test_safe_projector_checkpoint_roundtrip_and_integrity(tmp_path):
    projector = BridgeProjector((2, 4), (1, 6), "mlp", rank=8)
    path = tmp_path / "map"; save_projector(projector, path, {"split": "train"})
    reloaded = load_projector(path)
    kv = KV(torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    torch.testing.assert_close(projector(kv).k, reloaded(kv).k)
    weights = path / "projector.safetensors"
    with weights.open("ab") as handle: handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash"): load_projector(path)


def test_mlp_fitting_runs_with_declared_budget():
    source = KV(torch.randn(64, 1, 4), torch.randn(64, 1, 4))
    target = KV(source.k * 0.5 + 0.1, source.v * 0.2 - 0.1)
    projector = BridgeProjector((1, 4), (1, 4), "mlp", rank=8)
    report = fit_mlp(projector, source, target, steps=12, batch_size=16)
    assert report["steps"] == 12 and report["row_presentations"] == 192
    assert report["last_loss"] >= 0


def test_control_plane_rejects_text_and_extra_fields():
    payload = {"run": "00000000-0000-0000-0000-000000000041", "op": 2, "epoch": 0}
    assert ControlEvent.parse(payload).op == Op.TICK
    with pytest.raises(ValueError): ControlEvent.parse(dict(payload, answer="secret"))
    with pytest.raises(ValueError): ControlEvent.parse(dict(payload, epoch=True))


def test_paired_statistics_and_validity_gate():
    result = paired_bootstrap([1.0] * 100, [0.0] * 100, repetitions=500)
    assert result["ci95"] == [1.0, 1.0]
    assert e2_gate(result, valid_channel_audit=True, preregistered=True) == "PASSED"
    assert e2_gate(result, valid_channel_audit=False, preregistered=True) == "INVALID"
    assert e2_gate(result, valid_channel_audit=True, preregistered=False) == "BLOCKED"
    result = paired_bootstrap([0.0] * 100, [0.0] * 100, repetitions=500)
    assert e2_gate(result, valid_channel_audit=True, preregistered=True) == "FAILED"


def test_energy_unmeasured_is_not_zero():
    ledger = CostLedger(); ledger.validate(); assert ledger.joules is None
    ledger.wire_bytes = -1
    with pytest.raises(ValueError): ledger.validate()


def test_unsupported_rotary_fails_closed():
    model = ToyModel(); model.config.rope_scaling = {"rope_type": "dynamic"}
    with pytest.raises(ValueError): FrozenDecoder(model)
