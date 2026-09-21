"""Make a frozen oMLX / mlx-vlm model differentiable w.r.t. injected cache entries WITHOUT editing the runtime's files.
Applied only inside Drift's own training / probing processes:
  - `mx.async_eval` is a scheduling hint inside forward passes (their comments: outputs are bit-identical); MLX forbids
    it inside a gradient transformation, so it becomes a no-op here;
  - index-producing ops (argpartition / argsort / argmax) return integers; when their inputs depend on the traced
    entries MLX tries to differentiate through the indices of the following gather and refuses. `stop_gradient` on an
    integer result changes nothing numerically and removes that request. The sparse selection itself stays as the model
    computes it; no gradient flows through the CHOICE of blocks, only through the chosen entries (straight-through in
    the usual sense for top-k routing)."""
from __future__ import annotations
import mlx.core as mx

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    mx.async_eval = lambda *args, **kwargs: None
    for name in ("argpartition", "argsort", "argmax", "argmin"):
        original = getattr(mx, name)
        setattr(mx, name, (lambda fn: lambda *a, **k: mx.stop_gradient(fn(*a, **k)))(original))
    _applied = True


def use_plain_rotary(model) -> int:
    """mlx-vlm's rotary modules apply M-RoPE through a fused Metal kernel (no backward pass) unless `fused_apply` is
    False; the plain-ops branch in the same module is numerically the same rotation. Returns how many were switched."""
    switched = 0
    for _, module in model.named_modules():
        if getattr(module, "fused_apply", None):
            module.fused_apply = False
            switched += 1
    return switched
