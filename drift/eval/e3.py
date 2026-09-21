"""E3 organism comparison (spec M2.4 / P1 E3).

Arms on the same task set: A alone, B alone, coupled pair (hive), and an external
text-pair arm (the Duo room) supplied as a callback so this module never talks to
OMP. Every arm records its declared budget and measured costs: local decode tokens,
mailbox tokens, foreign entries attended, wall-clock. Budgets cannot always be
matched exactly; the report carries both so the quality/cost trade-off is visible
instead of a single token-count match being presented as compute equality.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import torch
from drift.eval.generate import generate


@dataclass(frozen=True)
class E3Task:
    id: str
    prompt_a: torch.Tensor
    prompt_b: torch.Tensor


@dataclass(frozen=True)
class Budget:
    max_local_tokens: int
    max_wall_seconds: float


@dataclass
class ArmCost:
    local_tokens: int = 0
    mailbox_tokens: int = 0
    foreign_tokens_attended: int = 0
    wall_seconds: float = 0.0
    over_budget: bool = False


def _solo(adapter: Any, prompt: torch.Tensor, new_tokens: int, budget: Budget) -> tuple[list[int], ArmCost]:
    started = time.perf_counter()
    ids, consumed = generate(adapter, prompt, new_tokens)
    cost = ArmCost(local_tokens=consumed, wall_seconds=time.perf_counter() - started)
    cost.over_budget = cost.local_tokens > budget.max_local_tokens or cost.wall_seconds > budget.max_wall_seconds
    return ids, cost


def _coupled(controller, names: tuple[str, str], prompts: dict[str, torch.Tensor], epochs: int, budget: Budget) -> tuple[dict[str, list[int]], ArmCost]:
    """Both members prefill their prompts at epoch 0, then decode one token per epoch."""
    started = time.perf_counter()
    inputs = dict(prompts)
    generated = {name: [] for name in names}
    cost = ArmCost()
    with torch.no_grad():
        for _ in range(epochs):
            reports = controller.tick(inputs)
            for name in names:
                token = int(reports[name].output[-1].argmax())
                generated[name].append(token)
                cost.local_tokens += int(inputs[name].numel())
                cost.foreign_tokens_attended += reports[name].foreign_tokens
                cost.mailbox_tokens += controller.workers[name].counters.mail_tokens
                inputs[name] = torch.tensor([token], dtype=torch.long)
    cost.wall_seconds = time.perf_counter() - started
    cost.over_budget = cost.local_tokens > budget.max_local_tokens or cost.wall_seconds > budget.max_wall_seconds
    return generated, cost


def run_e3(adapter_a: Any, adapter_b: Any, make_controller: Callable[[], Any], names: tuple[str, str],
           tasks: Sequence[E3Task], new_tokens: int, budget: Budget,
           text_pair: Callable[[E3Task, Budget], tuple[list[int], ArmCost]] | None = None) -> list[dict]:
    results = []
    for task in tasks:
        a_ids, a_cost = _solo(adapter_a, task.prompt_a, new_tokens, budget)
        b_ids, b_cost = _solo(adapter_b, task.prompt_b, new_tokens, budget)
        controller = make_controller()
        coupled_ids, coupled_cost = _coupled(controller, names, {names[0]: task.prompt_a, names[1]: task.prompt_b}, new_tokens, budget)
        row = {"id": task.id, "budget": vars(budget),
               "solo_a": {"ids": a_ids, "cost": vars(a_cost)},
               "solo_b": {"ids": b_ids, "cost": vars(b_cost)},
               "coupled": {"ids": coupled_ids, "cost": vars(coupled_cost)}}
        if text_pair is not None:
            t_ids, t_cost = text_pair(task, budget)
            row["text_pair"] = {"ids": t_ids, "cost": vars(t_cost)}
        results.append(row)
    return results
