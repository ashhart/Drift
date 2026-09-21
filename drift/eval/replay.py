"""Counterfactual replay (spec M3.3, D9).

Fork from a checkpoint taken BEFORE the receiver first sees the candidate entries,
then run the same schedule under several conditions:

  active:   the original entries
  ablated:  the candidate entries removed from the receiver's mail bank before exposure
  wrong:    the candidate entries replaced by matched control rows (when supplied)

Everything exogenous (inputs, epochs, RNG) is held fixed; downstream model states
diverge. Outputs are returned for an EXTERNAL scorer; nothing here is fed back to
any worker. Removing an entry after it has already changed the receiver's native
state is not a clean control, which is why the fork point is a precondition.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import torch
from drift.core.pool import PoolBank, PoolView
from drift.core.types import KV
from drift.runtime.worker import StepReport


@dataclass(frozen=True)
class ReplayCondition:
    name: str
    remove_positions: frozenset[int] = frozenset()
    replace_rows: Mapping[int, KV | torch.Tensor] | None = None      # receiver layer -> rows aligned with removed positions


def filter_view(view: PoolView, keep: torch.Tensor) -> PoolView:
    layers = {}
    for layer, entry in view.layers.items():
        layers[layer] = KV(entry.k[keep], entry.v[keep]) if isinstance(entry, KV) else entry[keep]
    from types import MappingProxyType
    return PoolView(view.epoch, view.positions[keep], view.writers[keep], MappingProxyType(layers))


def apply_condition(bank: PoolBank, condition: ReplayCondition) -> None:
    """Edit a FORKED (restored) bank's pinned view in place for one replay branch."""
    view = bank.pin()
    if view is None or not condition.remove_positions:
        return
    removed = torch.tensor([int(p) in condition.remove_positions for p in view.positions.tolist()])
    if condition.replace_rows is None:
        bank._view = filter_view(view, ~removed)
        return
    layers = {}
    for layer, entry in view.layers.items():
        replacement = condition.replace_rows[layer]
        if isinstance(entry, KV):
            k, v = entry.k.clone(), entry.v.clone()
            k[removed], v[removed] = replacement.k, replacement.v
            layers[layer] = KV(k, v)
        else:
            rows = entry.clone()
            rows[removed] = replacement
            layers[layer] = rows
    from types import MappingProxyType
    bank._view = PoolView(view.epoch, view.positions, view.writers, MappingProxyType(layers))


def replay(controller, checkpoint: dict, receiver: str, mail_bank_of: Callable[[str], PoolBank],
           conditions: Sequence[ReplayCondition], schedule: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, list[StepReport]]:
    """Run `schedule` from `checkpoint` once per condition; return the receiver's reports."""
    results: dict[str, list[StepReport]] = {}
    for condition in conditions:
        controller.restore(checkpoint)
        apply_condition(mail_bank_of(receiver), condition)
        reports = []
        with torch.no_grad():
            for inputs in schedule:
                reports.append(controller.tick(inputs)[receiver])
        results[condition.name] = reports
    controller.restore(checkpoint)
    return results
