"""E3 harness, competition harness and registry on dense toys / synthetic data."""
from __future__ import annotations
from pathlib import Path
from uuid import UUID
import pytest
import torch
from drift.adapters.toy import DenseAdapter
from drift.compete.harness import Invariants, Limits, Policy, Submission, run_submission, scoreboard
from drift.core.pool import PoolBank
from drift.eval.e3 import ArmCost, Budget, E3Task, run_e3
from drift.registry import ModelEntry, read_entry, usable_for, write_entry
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.hive import HiveController
from drift.runtime.toy import ToyModel
from drift.runtime.worker import Worker
from drift.translate.pool import Layout, PoolFormat, Translator
from drift.transport.wire2 import FrameCodec2

SESSION = UUID(int=77)
FMT = PoolFormat("pool.v1", 3, 8, "aa" * 32)


def adapters():
    torch.manual_seed(0)
    return {"A": DenseAdapter(FrozenDecoder(ToyModel(seed=1, kvheads=2, dim=6, layers=3))),
            "B": DenseAdapter(FrozenDecoder(ToyModel(seed=2, kvheads=1, dim=4, layers=3)))}


def make_controller(ads):
    workers = {}
    ids = {"A": 1, "B": 2}
    for name, adapter in ads.items():
        d = adapter.descriptor
        t = Translator(name, Layout("kv_split", d.kv_heads, d.head_dim), {l: l for l in d.kv_layers}, FMT)
        with torch.no_grad():
            for p in t.parameters():
                p.mul_(0.3)
        workers[name] = Worker(name, ids[name], SESSION, adapter, t, PoolBank(SESSION, t, ids[name], {v for k, v in ids.items() if k != name}), override=0.5)
    return HiveController(workers, FrameCodec2(b"e" * 32))


def test_e3_reports_all_arms_with_costs_and_budgets():
    ads = adapters()
    tasks = [E3Task("t1", torch.tensor([1, 2, 3]), torch.tensor([4, 5])), E3Task("t2", torch.tensor([6]), torch.tensor([7, 8, 9]))]
    budget = Budget(max_local_tokens=8, max_wall_seconds=60)
    rows = run_e3(ads["A"], ads["B"], lambda: make_controller(ads), ("A", "B"), tasks, new_tokens=3, budget=budget,
                  text_pair=lambda task, b: ([0, 0, 0], ArmCost(local_tokens=99, wall_seconds=0.0, over_budget=True)))
    assert [r["id"] for r in rows] == ["t1", "t2"]
    r = rows[0]
    assert len(r["solo_a"]["ids"]) == 3 and r["solo_a"]["cost"]["local_tokens"] == 3 + 2
    assert set(r["coupled"]["ids"]) == {"A", "B"} and r["coupled"]["cost"]["foreign_tokens_attended"] > 0
    assert r["text_pair"]["cost"]["over_budget"] is True and r["budget"]["max_local_tokens"] == 8


def test_competition_verifies_signatures_invariants_and_never_ranks_invalid(tmp_path):
    ads = adapters()
    key = b"k" * 32
    limits = Limits(max_pool_width=8, max_training_rows=300, max_wall_seconds=60)
    invariants = Invariants({name: a.frozen_digest() for name, a in ads.items()})
    layouts = {n: Layout("kv_split", a.descriptor.kv_heads, a.descriptor.head_dim) for n, a in ads.items()}
    level_maps = {n: {l: l for l in a.descriptor.kv_layers} for n, a in ads.items()}
    g = torch.Generator().manual_seed(1)
    dev_rows = {lvl: {n: torch.randn(300, layouts[n].width, generator=g) for n in ads} for lvl in range(3)}

    def evaluate(translators, gates):
        return 0.5 + 0.01 * translators["A"].pool.width          # stands in for a held-out score

    good = Submission.sign("alice", Policy("ridge", 8, 6, -6.0, "one_epoch_lag", 8, 8, 2, 200), key)
    ok = run_submission(good, key, limits, invariants, ads, layouts, level_maps, dev_rows, evaluate, 0.55, tmp_path)
    assert ok.verdict == "PASSED" and ok.quality == pytest.approx(0.56) and (tmp_path / "alice.json").exists()
    weaker = run_submission(Submission.sign("bob", Policy("ridge", 8, 2, -6.0, "one_epoch_lag", 8, 8, 2, 200), key),
                            key, limits, invariants, ads, layouts, level_maps, dev_rows, evaluate, 0.55, tmp_path)
    assert weaker.verdict == "FAILED"
    forged_result = run_submission(Submission("mallory", good.policy, good.manifest_sha256, "00" * 32),
                                   key, limits, invariants, ads, layouts, level_maps, dev_rows, evaluate, 0.55, tmp_path)
    assert forged_result.verdict == "INVALID"
    abi = Submission.sign("carol", Policy("magic", 8, 6, -6.0, "one_epoch_lag", 8, 8, 2, 200), key)
    assert run_submission(abi, key, limits, invariants, ads, layouts, level_maps, dev_rows, evaluate, 0.55, tmp_path).verdict == "INVALID"
    over = Submission.sign("dave", Policy("ridge", 8, 6, -6.0, "one_epoch_lag", 8, 8, 2, 10_000), key)
    assert run_submission(over, key, limits, invariants, ads, layouts, level_maps, dev_rows, evaluate, 0.55, tmp_path).verdict == "INVALID"
    # Touching a backbone makes the run INVALID even with a valid submission.
    with torch.no_grad():
        next(ads["A"].decoder.model.parameters()).add_(1.0)
    tampered = run_submission(good, key, limits, invariants, ads, layouts, level_maps, dev_rows, evaluate, 0.55, tmp_path)
    assert tampered.verdict == "INVALID" and "backbone" in tampered.reason
    board = scoreboard([ok, weaker, forged_result, tampered])
    assert [row["competitor"] for row in board[:2]] == ["alice", "bob"] and board[0]["rank"] == 1
    assert all(row["rank"] is None for row in board[2:])


def test_registry_entries_require_evidence(tmp_path):
    entry = ModelEntry("qwen3.8-flash-next-tiny", "qwen4_exp/torch", None, None, None, None, None, "ab" * 32,
                       "none", "macbook", 1, "cd" * 32)
    path = write_entry(tmp_path, entry)
    assert path.exists() and read_entry(tmp_path, entry.model_id) == entry
    assert usable_for(entry, "tiny") and not usable_for(entry, "M0")
    with pytest.raises(ValueError):
        ModelEntry("x", "unknown/torch", None, None, None, None, None, "ab" * 32, "none", "h", 0, None).check()
    with pytest.raises(ValueError):
        ModelEntry("x", "glm5_next/torch", None, None, None, None, None, "ab" * 32, "none", "h", 1, None).check()
    with pytest.raises(ValueError):
        ModelEntry("x", "glm5_next/torch", "r", "rev", None, None, None, "ab" * 32, "nvfp4", "h", 2, "cd" * 32).check()
