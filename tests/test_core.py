from dataclasses import replace
from uuid import UUID
import pytest
import torch
from drift.core.types import Delta, KV
from drift.core.position import basic_rope, recency_positions
from drift.core.attention import Gate, attend
from drift.core.projector import BridgeProjector
from drift.core.memory import ForeignKVBank

SESSION = UUID(int=41)


def identity(heads=2, dim=4):
    projector = BridgeProjector((heads, dim), (heads, dim))
    with torch.no_grad():
        for sub in (projector.k_map, projector.v_map):
            sub.map.weight.copy_(torch.eye(heads * dim)); sub.map.bias.zero_()
    return projector


def sample(tokens=3, heads=2, dim=4):
    return KV(torch.randn(tokens, heads, dim), torch.randn(tokens, heads, dim))


def test_canonical_type_and_nan_rejection():
    sample().check()
    kv = sample(); kv.k[0, 0, 0] = float("nan")
    with pytest.raises(ValueError): kv.check()
    with pytest.raises(ValueError): KV(torch.ones(2, 3), torch.ones(2, 3)).check()


def test_rotary_inverse_and_relative_positions():
    kv = sample()
    positions = torch.tensor([2, 5, 9])
    recovered = basic_rope(basic_rope(kv.k, positions), -positions)
    torch.testing.assert_close(recovered, kv.k)
    assert recency_positions(positions, 1).tolist() == [-7, -4, 0]
    with pytest.raises(ValueError): recency_positions(torch.tensor([0, 0]), 1)


def test_hard_closed_gate_is_exact_native_path():
    q, native, foreign = torch.randn(2, 4, 4), sample(5), sample(3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    baseline, _ = attend(q, native, mask)
    foreign.k.fill_(1000)
    closed, mass = attend(q, native, mask, foreign, override=0.0)
    assert torch.equal(baseline, closed) and torch.count_nonzero(mass) == 0


def test_foreign_prior_is_inside_softmax():
    q = torch.zeros(1, 2, 4)
    native = KV(torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
    foreign = KV(torch.zeros(1, 1, 4), torch.full((1, 1, 4), 3.0))
    output, mass = attend(q, native, torch.ones(1, 1, dtype=torch.bool), foreign, override=0.5)
    torch.testing.assert_close(mass, torch.full((1, 2), 1/3))
    torch.testing.assert_close(output, torch.full((1, 2, 4), 5/3))


def test_gate_and_projector_gradients_exist():
    q, native, foreign, gate = torch.randn(1, 4, 4), sample(3), sample(2), Gate()
    foreign.k.requires_grad_(True); foreign.v.requires_grad_(True)
    output, _ = attend(q, native, torch.ones(1, 3, dtype=torch.bool), foreign, gate=gate)
    output.square().sum().backward()
    assert gate.logit.grad is not None and gate.logit.grad.abs() > 0
    assert foreign.k.grad is not None and foreign.v.grad is not None


def test_masked_foreign_is_native_only():
    q, native, foreign = torch.randn(1, 4, 4), sample(3), sample(2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected, _ = attend(q, native, mask)
    actual, _ = attend(q, native, mask, foreign, override=1,
                       allowed_foreign=torch.zeros(1, 2, dtype=torch.bool))
    assert torch.equal(expected, actual)


def test_causal_future_key_has_no_effect():
    q, native = torch.randn(2, 4, 4), sample(2)
    mask = torch.tensor([[True, False], [True, True]])
    original, _ = attend(q, native, mask)
    changed = native.clone(); changed.k[1].fill_(999); changed.v[1].fill_(999)
    after, _ = attend(q, changed, mask)
    torch.testing.assert_close(original[0], after[0])


@pytest.mark.parametrize("override", [-1.0, 1.1])
def test_bad_gate_rejected(override):
    with pytest.raises(ValueError):
        attend(torch.randn(1, 4, 4), sample(), torch.ones(1, 3, dtype=torch.bool), override=override)


def test_ridge_recovers_heldout_affine_maps():
    source = sample(200, 2, 4)
    oracle_k, oracle_v = torch.randn(8, 6), torch.randn(8, 6)
    target = KV((source.k.flatten(1) @ oracle_k + 0.4).reshape(200, 1, 6),
                (source.v.flatten(1) @ oracle_v - 0.3).reshape(200, 1, 6))
    projector = BridgeProjector((2, 4), (1, 6))
    projector.fit(source, target, ridge=1e-6)
    heldout = sample(17, 2, 4); predicted = projector(heldout)
    torch.testing.assert_close(predicted.k.flatten(1), heldout.k.flatten(1) @ oracle_k + 0.4)
    torch.testing.assert_close(predicted.v.flatten(1), heldout.v.flatten(1) @ oracle_v - 0.3)


def test_bank_projects_once_preserves_sinks_and_pinned_old_view():
    projector = identity()
    bank = ForeignKVBank(SESSION, 0, [(0, 1, projector)], sinks=2, recent=3)
    first = Delta(SESSION, 0, 0, 0, 0, {0: sample(5)})
    bank.commit(first)
    old = bank.pin(); old_values = old.layers[1].k.clone()
    first.layers[0].k.fill_(777)  # sender mutation cannot change received view.
    assert torch.equal(old.layers[1].k, old_values)
    bank.commit(Delta(SESSION, 0, 1, 1, 5, {0: sample(4)}))
    assert bank.pin().positions.tolist() == [0, 1, 6, 7, 8]
    assert old.positions.tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(old.layers[1].k, old_values)
    assert bank.pin(now_epoch=20, max_age=3) is None


@pytest.mark.parametrize("mutation", ["session", "direction", "sequence", "start", "epoch", "layers"])
def test_bad_publication_does_not_change_bank(mutation):
    bank = ForeignKVBank(SESSION, 0, [(0, 1, identity())])
    bank.commit(Delta(SESSION, 0, 0, 0, 0, {0: sample(3)}))
    good = Delta(SESSION, 0, 1, 1, 3, {0: sample(2)})
    values = {"session": UUID(int=99), "direction": 1, "sequence": 0,
              "start": 4, "epoch": 0, "layers": {1: sample(2)}}
    old = bank.pin()
    with pytest.raises(ValueError): bank.commit(replace(good, **{mutation: values[mutation]}))
    assert bank.pin() is old and bank.next_sequence == 1


def test_atomic_layer_set_validation():
    bank = ForeignKVBank(SESSION, 0, [(0, 0, identity()), (2, 2, identity())])
    with pytest.raises(ValueError): bank.commit(Delta(SESSION, 0, 0, 0, 0, {0: sample()}))
    assert bank.pin() is None
