"""Greedy diagnostic generation for any adapter. Counts every local token."""
from __future__ import annotations
from typing import Any
import torch
from drift.adapters.base import ForeignEntries


@torch.no_grad()
def generate(adapter: Any, prompt_ids: torch.Tensor, max_new_tokens: int,
             foreign: ForeignEntries | None = None, override: float | None = 1.0,
             gates=None, eos_id: int | None = None) -> tuple[list[int], int]:
    """Returns (generated ids, local tokens consumed including the prompt)."""
    if max_new_tokens <= 0:
        raise ValueError("positive generation budget required")
    out = adapter.forward(prompt_ids, foreign=foreign, gates=gates, override=override)
    consumed, generated = int(prompt_ids.numel()), []
    for step in range(max_new_tokens):
        token = int(out.output[-1].argmax())
        generated.append(token)
        if token == eos_id or step == max_new_tokens - 1:
            break
        out = adapter.forward(torch.tensor([token], dtype=torch.long), out.state, foreign=foreign,
                              gates=gates, override=override)
        consumed += 1
    return generated, consumed
