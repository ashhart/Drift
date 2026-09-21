from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .types import KV


class Gate(nn.Module):
    def __init__(self, initial_logit: float = -6.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(initial_logit)))


def expand_gqa(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    if x.ndim != 3 or query_heads % x.shape[1]:
        raise ValueError("receiver query heads must be a multiple of receiver KV heads")
    # This is receiver-native GQA expansion, NOT a cross-family projector.
    return x.repeat_interleave(query_heads // x.shape[1], dim=1)


def attend(q: torch.Tensor, native_rotated: KV, allowed_native: torch.Tensor,
           foreign_rotated: KV | None = None, gate: Gate | None = None,
           override: float | None = None,
           allowed_foreign: torch.Tensor | None = None,
           scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager, batch-one attention with foreign PRIOR inside the shared softmax.

    q:[Q,Hq,D], KV:[T,Hkv,D], boolean masks:[Q,T], True=allowed.
    override=0 is an EXACT native-only path; override=1 removes gate suppression.
    A zero value tensor is not a valid ablation: its logits would still compete.
    Returned foreign mass:[Q,Hq], diagnostic only, never proof of incorporation.
    """
    if q.ndim != 3 or not torch.isfinite(q).all():
        raise ValueError("invalid query")
    native_rotated.check()
    nq, hq, d = q.shape
    if allowed_native.dtype != torch.bool or allowed_native.shape != (nq, native_rotated.tokens):
        raise ValueError("native mask must be boolean [Q,T]")
    if not bool(allowed_native.any(dim=-1).all()):
        raise ValueError("each query needs at least one permitted native key")
    if native_rotated.k.shape[-1] != d:
        raise ValueError("native head dimension mismatch")
    kn = expand_gqa(native_rotated.k, hq)
    vn = expand_gqa(native_rotated.v, hq)
    multiplier = (1 / math.sqrt(d)) if scale is None else scale
    native_scores = torch.einsum("qhd,thd->hqt", q, kn).float() * multiplier
    native_scores = native_scores.masked_fill(~allowed_native[None], -torch.inf)
    use_foreign = foreign_rotated is not None and override != 0.0
    if override is not None and not 0.0 <= override <= 1.0:
        raise ValueError("gate override must be in [0,1]")
    if use_foreign:
        foreign_rotated.check()
        if foreign_rotated.k.shape[1:] != native_rotated.k.shape[1:]:
            raise ValueError("project foreign KV into RECEIVER KV shape first")
        nf = foreign_rotated.tokens
        if allowed_foreign is None:
            allowed_foreign = torch.ones((nq, nf), dtype=torch.bool, device=q.device)
        if allowed_foreign.dtype != torch.bool or allowed_foreign.shape != (nq, nf):
            raise ValueError("foreign mask must be boolean [Q,Tf]")
        if not bool(allowed_foreign.any()):
            use_foreign = False
    if not use_foreign:
        weights = native_scores.softmax(dim=-1).to(q.dtype)
        output = torch.einsum("hqt,thd->qhd", weights, vn)
        return output, torch.zeros((nq, hq), device=q.device, dtype=q.dtype)
    if override is None and gate is None:
        raise ValueError("foreign attention requires a gate or explicit override")
    log_prior = (math.log(override) if override is not None
                 else F.logsigmoid(gate.logit).float())
    kf = expand_gqa(foreign_rotated.k, hq)
    vf = expand_gqa(foreign_rotated.v, hq)
    foreign_scores = torch.einsum("qhd,thd->hqt", q, kf).float() * multiplier + log_prior
    foreign_scores = foreign_scores.masked_fill(~allowed_foreign[None], -torch.inf)
    weights = torch.cat((native_scores, foreign_scores), dim=-1).softmax(-1).to(q.dtype)
    nnative = native_rotated.tokens
    output = (torch.einsum("hqt,thd->qhd", weights[..., :nnative], vn)
              + torch.einsum("hqt,thd->qhd", weights[..., nnative:], vf))
    return output, weights[..., nnative:].sum(-1).transpose(0, 1)
