"""Hive-stage experiments (docs/HIVE_MIND.md §7): pool handoff, swap-in, writer departure.

These produce predictions for an EXTERNAL scorer. Nothing here sees answers.

H-E1 pool handoff:   writer A fills the pool; reader B answers from the pool only.
                     Arms: floor (question), ceiling (B rereads text), pool, hard_off,
                     direct (the pairwise Drift map on the same rows: the baseline).
H-E2 swap-in:        a newcomer enrolled AFTER the pool was written answers from it.
H-E3 departure:      a writer's entries stay readable after the writer is gone.
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
class PoolUnit:
    id: str
    writer_context: torch.Tensor          # ids in the writer's tokenizer
    reader_context: torch.Tensor          # same text in the reader's tokenizer (ceiling only)
    question: torch.Tensor                # reader tokenizer


def _detached(canonical: Mapping[int, KV | torch.Tensor]) -> dict[int, KV | torch.Tensor]:
    return {k: (v.clone(detach=True) if isinstance(v, KV) else v.detach()) for k, v in canonical.items()}


@torch.no_grad()
def write_pool(writer_adapter: Any, writer: Translator, context: torch.Tensor) -> dict[int, torch.Tensor]:
    return {level: rows.detach() for level, rows in writer.write(_detached(writer_adapter.forward(context).canonical)).items()}


@torch.no_grad()
def read_pool(reader: Translator, pool: Mapping[int, torch.Tensor], tokens: int) -> ForeignEntries:
    return ForeignEntries(torch.arange(tokens), reader.read(pool))


def run_pool_handoff(writer_adapter: Any, reader_adapter: Any, writer: Translator, reader: Translator,
                     units: Sequence[PoolUnit], new_tokens: int = 16, override: float = 1.0,
                     direct: Translator | None = None, eos_id: int | None = None) -> list[dict]:
    """`direct` is an optional pairwise map (writer canonical -> reader canonical) used as the
    Drift baseline; it is a Translator whose pool format is the writer's native layout."""
    results = []
    for unit in units:
        pool = write_pool(writer_adapter, writer, unit.writer_context)
        entries = read_pool(reader, pool, unit.writer_context.numel())
        arms = {
            "floor": generate(reader_adapter, unit.question, new_tokens, eos_id=eos_id),
            "ceiling": generate(reader_adapter, torch.cat((unit.reader_context, unit.question)), new_tokens, eos_id=eos_id),
            "pool": generate(reader_adapter, unit.question, new_tokens, foreign=entries, override=override, eos_id=eos_id),
            "hard_off": generate(reader_adapter, unit.question, new_tokens, foreign=entries, override=0.0, eos_id=eos_id),
        }
        if direct is not None:
            with torch.no_grad():
                canonical = _detached(writer_adapter.forward(unit.writer_context).canonical)
                direct_entries = ForeignEntries(torch.arange(unit.writer_context.numel()), direct.read(direct.write(canonical)))
            arms["direct"] = generate(reader_adapter, unit.question, new_tokens, foreign=direct_entries, override=override, eos_id=eos_id)
        results.append({"id": unit.id, **{f"{arm}_ids": ids for arm, (ids, _) in arms.items()},
                        "local_tokens": {arm: consumed for arm, (_, consumed) in arms.items()},
                        "pool_rows": int(unit.writer_context.numel())})
    return results


def run_swap_in(writer_adapter: Any, writer: Translator, newcomer_adapter: Any, newcomer: Translator,
                units: Sequence[PoolUnit], new_tokens: int = 16, override: float = 1.0, eos_id: int | None = None) -> list[dict]:
    """The newcomer's translator must have been enrolled against the frozen pool format
    (same fingerprint) without touching the writer's translator. Arms: floor, ceiling, pool, hard_off."""
    if newcomer.pool.fingerprint != writer.pool.fingerprint:
        raise ValueError("newcomer is not enrolled in this pool format")
    return run_pool_handoff(writer_adapter, newcomer_adapter, writer, newcomer, units, new_tokens, override, None, eos_id)


def run_departure(pool_rows: Mapping[int, torch.Tensor], tokens: int, remaining_adapter: Any, remaining: Translator,
                  question: torch.Tensor, new_tokens: int = 16, override: float = 1.0, eos_id: int | None = None) -> dict:
    """Given pool rows a departed writer left behind, the remaining member reads them with no
    writer process alive. Arms: floor, pool, hard_off."""
    entries = read_pool(remaining, pool_rows, tokens)
    arms = {
        "floor": generate(remaining_adapter, question, new_tokens, eos_id=eos_id),
        "pool": generate(remaining_adapter, question, new_tokens, foreign=entries, override=override, eos_id=eos_id),
        "hard_off": generate(remaining_adapter, question, new_tokens, foreign=entries, override=0.0, eos_id=eos_id),
    }
    return {**{f"{arm}_ids": ids for arm, (ids, _) in arms.items()}, "local_tokens": {arm: c for arm, (_, c) in arms.items()}}
