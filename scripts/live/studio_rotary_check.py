"""Studio: is mlx-vlm's plain-ops M-RoPE path the same rotation as its fused kernel (and as Drift's Rope) for text
positions given as [1, T] and as [3, 1, T]? No model weights are needed for the rotary module itself, but the module
is taken from the loaded model so that style / sections / scaling are the real ones."""
import json
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope
from drift.serving.studio_guard import acquire
acquire("studio_rotary_check.py", need_gb=140)
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
rotary = next(m for _, m in model.named_modules() if getattr(m, "fused_apply", None) is not None)
print("style", rotary.style, "| pairing", getattr(rotary, "pairing", None), "| dim", rotary.dim, "| sections", rotary.mrope_section, "| scaling", rotary.attention_scaling)
mx.random.seed(0)
T = 9
q, k = mx.random.normal((1, 24, T, 256)), mx.random.normal((1, 2, T, 256))
pos2 = mx.array(np.arange(100, 100 + T, dtype=np.int32))[None]
pos3 = mx.broadcast_to(pos2[None], (3, 1, T))
out = {}
for label, pos in (("2d", pos2), ("3d", pos3)):
    rotary.fused_apply = True; fq, fk = rotary.apply_rotary(q, k, pos)
    rotary.fused_apply = False; pq, pk = rotary.apply_rotary(q, k, pos)
    mx.eval(fq, fk, pq, pk)
    out[label] = {"plain_vs_fused_q": float(mx.abs(fq.astype(mx.float32) - pq.astype(mx.float32)).max()), "plain_vs_fused_k": float(mx.abs(fk.astype(mx.float32) - pk.astype(mx.float32)).max())}
    mine = Rope(10_000_000.0, 64).apply(k, np.arange(100, 100 + T))
    out[label]["drift_rope_vs_fused_k"] = float(mx.abs(mine - fk.astype(mx.float32)).max()); out[label]["drift_rope_vs_plain_k"] = float(mx.abs(mine - pk.astype(mx.float32)).max())
print(json.dumps(out, indent=1))
