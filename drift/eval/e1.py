"""E1 handoff diagnostic for any adapter pair (spec M1.3 / P1 E1).

Arms per unit (fields: id, context ids for the donor, context ids for the receiver,
question ids for the receiver):
  floor:        receiver answers from the question alone
  ceiling:      receiver rereads its own context, then the question
  foreign:      donor reads the context; receiver gets the translated entries + question
  hard_off:     same as foreign with override 0 (must equal floor exactly)
  wrong_context: entries from a different unit's context (leakage control)

Answers are never present here; scoring happens outside (score_e1). Publications
go through the writer/reader pair, never directly into a bank.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import torch
from drift.adapters.base import ForeignEntries
from drift.core.types import KV
from drift.eval.generate import generate
from drift.translate.pool import Translator


@dataclass(frozen=True)
class E1Unit:
    id: str
    donor_context: torch.Tensor
    receiver_context: torch.Tensor
    question: torch.Tensor


def _entries(donor: Any, writer: Translator, reader: Translator, context: torch.Tensor) -> ForeignEntries:
    with torch.no_grad():
        canonical = donor.forward(context).canonical
    canonical = {k: (v.clone(detach=True) if isinstance(v, KV) else v.detach()) for k, v in canonical.items()}
    with torch.no_grad():
        entries = reader.read(writer.write(canonical))
    return ForeignEntries(torch.arange(context.numel()), entries)


def run_e1(donor: Any, receiver: Any, writer: Translator, reader: Translator, units: Sequence[E1Unit],
           new_tokens: int = 16, override: float = 1.0, eos_id: int | None = None) -> list[dict]:
    if len(units) < 2:
        raise ValueError("wrong-context control needs at least two units")
    results = []
    for index, unit in enumerate(units):
        other = units[(index + 1) % len(units)]
        entries = _entries(donor, writer, reader, unit.donor_context)
        wrong = _entries(donor, writer, reader, other.donor_context)
        arms = {
            "floor": generate(receiver, unit.question, new_tokens, eos_id=eos_id),
            "ceiling": generate(receiver, torch.cat((unit.receiver_context, unit.question)), new_tokens, eos_id=eos_id),
            "foreign": generate(receiver, unit.question, new_tokens, foreign=entries, override=override, eos_id=eos_id),
            "hard_off": generate(receiver, unit.question, new_tokens, foreign=entries, override=0.0, eos_id=eos_id),
            "wrong_context": generate(receiver, unit.question, new_tokens, foreign=wrong, override=override, eos_id=eos_id),
        }
        results.append({"id": unit.id,
                        **{f"{arm}_ids": ids for arm, (ids, _) in arms.items()},
                        "local_tokens": {arm: consumed for arm, (_, consumed) in arms.items()},
                        "foreign_tokens": int(entries.positions.numel())})
    return results
