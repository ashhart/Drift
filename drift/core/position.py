from __future__ import annotations
import torch


def apply_cos_sin(x: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE. x:[T,H,D], cos/sin:[T,D]. No value rotation.

    This function is NOT the receiver's frequency policy. Real adapters obtain
    cos/sin from the checkpoint's own rotary module, including its scaling.
    """
    if x.ndim != 3 or x.shape[-1] % 2 or cos.shape != (x.shape[0], x.shape[-1]):
        raise ValueError("bad RoPE shape or odd rotary dimension")
    if sin.shape != cos.shape:
        raise ValueError("cos/sin mismatch")
    half = x.shape[-1] // 2
    quarter_turn = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[:, None, :] + quarter_turn * sin[:, None, :]


def basic_rope(x: torch.Tensor, positions: torch.Tensor,
               theta: float = 10000.0) -> torch.Tensor:
    """Toy/default RoPE only; do not substitute this for a Llama-3.1 adapter."""
    if positions.shape != (x.shape[0],) or theta <= 1:
        raise ValueError("invalid position vector or base")
    freq = theta ** (-torch.arange(0, x.shape[-1], 2,
                                  device=x.device, dtype=torch.float32) / x.shape[-1])
    phase = positions.to(x.device).float()[:, None] * freq[None, :]
    phase = torch.cat((phase, phase), dim=-1)
    return apply_cos_sin(x, phase.cos().to(x.dtype), phase.sin().to(x.dtype))


def recency_positions(source_positions: torch.Tensor,
                      first_query_position: int) -> torch.Tensor:
    """Proposed foreign-memory policy: newest entry is one slot before query.

    Preserve SOURCE token distances, not semantic/tokenizer alignment. Negative
    virtual positions are intentional. A runtime must validate this policy.
    Source positions stay ordered; sinks retain their original distances.
    """
    if source_positions.ndim != 1 or source_positions.numel() == 0:
        raise ValueError("positions must be a nonempty vector")
    if not bool(torch.all(source_positions[1:] > source_positions[:-1])):
        raise ValueError("source positions must be strictly increasing")
    return first_query_position - 1 - (source_positions[-1] - source_positions)
