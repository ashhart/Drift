"""E2 drift harness (spec M1.3, D20). Secrets are assigned only to the donor; the
receiver gets its question and authorized entries. Conditions:

  no_context, foreign, hard_off, wrong_secret (another unit's secret), time_shuffled
  (foreign positions permuted), random_map (an unfitted translator pair), oracle
  (receiver sees the secret text: the ceiling and a leakage sanity check).

Assignment is randomized before the trial. Predictions carry no answers; scoring is
external and paired at secret level. The gate helper is `drift.eval.metrics.e2_gate`.
"""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from drift.adapters.base import ForeignEntries
from drift.core.types import KV
from drift.eval.generate import generate
from drift.translate.pool import Translator


@dataclass(frozen=True)
class E2Unit:
    id: str
    secret_donor: torch.Tensor          # secret in the donor tokenizer
    secret_receiver: torch.Tensor       # same secret in the receiver tokenizer (oracle arm only)
    question: torch.Tensor


def _entries(donor, writer, reader, ids):
    with torch.no_grad():
        canonical = donor.forward(ids).canonical
        canonical = {k: (v.clone(detach=True) if isinstance(v, KV) else v.detach()) for k, v in canonical.items()}
        return ForeignEntries(torch.arange(ids.numel()), reader.read(writer.write(canonical)))


def _random_pair(writer: Translator, reader: Translator, seed: int) -> tuple[Translator, Translator]:
    w, r = copy.deepcopy(writer), copy.deepcopy(reader)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(w.parameters()) + list(r.parameters()):
            p.copy_(torch.randn(p.shape, generator=g) * p.std().clamp_min(1e-3))
    return w, r


def run_e2(donor: Any, receiver: Any, writer: Translator, reader: Translator, units: Sequence[E2Unit],
           new_tokens: int = 16, override: float = 1.0, seed: int = 0, eos_id: int | None = None) -> dict:
    if len(units) < 2:
        raise ValueError("wrong-secret control needs at least two units")
    rng = random.Random(seed)
    order = list(range(len(units)))
    rng.shuffle(order)
    random_writer, random_reader = _random_pair(writer, reader, seed)
    results = []
    for k, index in enumerate(order):
        unit = units[index]
        other = units[order[(k + 1) % len(order)]]
        entries = _entries(donor, writer, reader, unit.secret_donor)
        perm = torch.randperm(entries.positions.numel(), generator=torch.Generator().manual_seed(seed + k))
        shuffled = ForeignEntries(entries.positions, {
            layer: (KV(e.k[perm], e.v[perm]) if isinstance(e, KV) else e[perm]) for layer, e in entries.layers.items()})
        arms = {
            "no_context": generate(receiver, unit.question, new_tokens, eos_id=eos_id),
            "foreign": generate(receiver, unit.question, new_tokens, foreign=entries, override=override, eos_id=eos_id),
            "hard_off": generate(receiver, unit.question, new_tokens, foreign=entries, override=0.0, eos_id=eos_id),
            "wrong_secret": generate(receiver, unit.question, new_tokens,
                                     foreign=_entries(donor, writer, reader, other.secret_donor), override=override, eos_id=eos_id),
            "time_shuffled": generate(receiver, unit.question, new_tokens, foreign=shuffled, override=override, eos_id=eos_id),
            "random_map": generate(receiver, unit.question, new_tokens,
                                   foreign=_entries(donor, random_writer, random_reader, unit.secret_donor), override=override, eos_id=eos_id),
            "oracle": generate(receiver, torch.cat((unit.secret_receiver, unit.question)), new_tokens, eos_id=eos_id),
        }
        results.append({"id": unit.id, "trial_order": k,
                        **{f"{arm}_ids": ids for arm, (ids, _) in arms.items()},
                        "local_tokens": {arm: consumed for arm, (_, consumed) in arms.items()}})
    return {"seed": seed, "assignment_order": [units[i].id for i in order], "units": results}
