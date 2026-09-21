"""M1 mechanics on dense toys: behavior training, generation, E1 and E2 harnesses."""
from __future__ import annotations
import pytest
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.attention import Gate
from drift.eval.e1 import E1Unit, run_e1
from drift.eval.e2 import E2Unit, run_e2
from drift.eval.generate import generate
from drift.eval.metrics import e2_gate, paired_bootstrap
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.train.behavior import BehaviorExample, Preregistration, train_behavior
from drift.translate.pool import Layout, PoolFormat, Translator

FMT = PoolFormat("pool.v1", 3, 8, "cc" * 32)


def pair():
    torch.manual_seed(0)
    donor = DenseAdapter(FrozenDecoder(ToyModel(seed=1, kvheads=2, dim=6, layers=3)))
    receiver = DenseAdapter(FrozenDecoder(ToyModel(seed=2, kvheads=1, dim=4, layers=3)))
    writer = Translator("donor", Layout("kv_split", 2, 6), {0: 0, 1: 1, 2: 2}, FMT, kind="mlp", rank=8)
    reader = Translator("receiver", Layout("kv_split", 1, 4), {0: 0, 1: 1, 2: 2}, FMT, kind="mlp", rank=8)
    with torch.no_grad():
        for p in list(writer.parameters()) + list(reader.parameters()):
            p.mul_(0.2)
    return donor, receiver, writer, reader


def test_behavior_training_moves_only_sidecars_and_reduces_loss():
    donor, receiver, writer, reader = pair()
    gates = {i: Gate(-2.0) for i in (0, 1, 2)}
    g = torch.Generator().manual_seed(3)
    examples = [BehaviorExample(torch.randint(1, 40, (5,), generator=g), torch.randint(1, 40, (5,), generator=g),
                                torch.randint(1, 40, (3,), generator=g), aligned_rows=[(1, 1), (4, 4)]) for _ in range(6)]
    prereg = Preregistration(lambda_reg=1.0, lambda_kl=1.0, steps=40, learning_rate=3e-2, seed=1)
    assert len(prereg.digest()) == 64
    before = {k: v.clone() for k, v in list(writer.state_dict().items())}
    ledger = train_behavior(donor, receiver, writer, reader, gates, examples, prereg)
    assert ledger.optimizer_steps == 40 and ledger.donor_prefill_tokens == 200
    first = sum(l["loss"] for l in ledger.losses[:5]) / 5
    last = sum(l["loss"] for l in ledger.losses[-5:]) / 5
    assert last < first
    assert any(not torch.equal(v, before[k]) for k, v in writer.state_dict().items())
    assert all(p.grad is None for p in receiver.decoder.model.parameters())
    assert all(p.grad is None for p in donor.decoder.model.parameters())
    with pytest.raises(ValueError):
        train_behavior(donor, receiver, writer, writer, gates, examples, prereg)


def test_generate_counts_tokens_and_hard_off_equals_floor():
    donor, receiver, writer, reader = pair()
    prompt = torch.tensor([3, 4, 5])
    ids, consumed = generate(receiver, prompt, 5)
    assert len(ids) == 5 and consumed == 3 + 4
    ids_eos, consumed_eos = generate(receiver, prompt, 5, eos_id=ids[0])
    assert ids_eos == [ids[0]] and consumed_eos == 3


def test_e1_arms_and_controls():
    donor, receiver, writer, reader = pair()
    g = torch.Generator().manual_seed(4)
    units = [E1Unit(f"u{i}", torch.randint(1, 40, (6,), generator=g), torch.randint(1, 40, (6,), generator=g),
                    torch.randint(1, 40, (2,), generator=g)) for i in range(3)]
    results = run_e1(donor, receiver, writer, reader, units, new_tokens=4)
    assert [r["id"] for r in results] == ["u0", "u1", "u2"]
    for r in results:
        assert r["hard_off_ids"] == r["floor_ids"]                    # exact native path
        assert r["foreign_tokens"] == 6 and set(r["local_tokens"]) == {"floor", "ceiling", "foreign", "hard_off", "wrong_context"}
        assert r["local_tokens"]["ceiling"] > r["local_tokens"]["floor"]
    assert any(r["foreign_ids"] != r["floor_ids"] for r in results)  # the path is live (random weights: no claim)
    with pytest.raises(ValueError):
        run_e1(donor, receiver, writer, reader, units[:1])


def test_e2_randomizes_assignment_and_runs_all_conditions():
    donor, receiver, writer, reader = pair()
    g = torch.Generator().manual_seed(5)
    units = [E2Unit(f"s{i}", torch.randint(1, 40, (5,), generator=g), torch.randint(1, 40, (5,), generator=g),
                    torch.randint(1, 40, (2,), generator=g)) for i in range(4)]
    out = run_e2(donor, receiver, writer, reader, units, new_tokens=3, seed=7)
    assert sorted(out["assignment_order"]) == ["s0", "s1", "s2", "s3"] and out["assignment_order"] != ["s0", "s1", "s2", "s3"]
    arms = {"no_context", "foreign", "hard_off", "wrong_secret", "time_shuffled", "random_map", "oracle"}
    for unit in out["units"]:
        assert {k[:-4] for k in unit if k.endswith("_ids")} == arms
        assert unit["hard_off_ids"] == unit["no_context_ids"]
    # The gate helper refuses an unaudited channel and an unpreregistered run.
    summary = paired_bootstrap([1, 0, 1, 1], [0, 0, 1, 0], repetitions=200)
    assert e2_gate(summary, valid_channel_audit=False, preregistered=True) == "INVALID"
    assert e2_gate(summary, valid_channel_audit=True, preregistered=False) == "BLOCKED"
    assert e2_gate(summary, valid_channel_audit=True, preregistered=True, minimum_units=4) in {"PASSED", "FAILED"}
