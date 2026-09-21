"""Studio, oMLX runtime: can a gradient w.r.t. injected cache entries flow through the frozen 4-bit Qwen3.8-Flash-Next?
(Prerequisite for answer-level training of the GLM -> Qwen translator.)"""
import json, time, traceback
import os
for _switch in ("OMLX_QWEN4_EAGER_DISPATCH", "OMLX_QWEN4_HC_FUSED", "OMLX_QWEN4_HC_HYBRID"):
    os.environ[_switch] = "0"              # oMLX's own switches: fused / eager paths have no backward pass; must be set before its modules import
import mlx.core as mx
KERNEL_CALLS = {}
_make_kernel = mx.fast.metal_kernel


def _logged_kernel(*a, **k):                                       # diagnostic: which custom Metal kernels are on the forward path?
    kernel, name = _make_kernel(*a, **k), k.get("name", a[0] if a else "?")
    def call(*ca, **ck):
        KERNEL_CALLS[name] = KERNEL_CALLS.get(name, 0) + 1
        return kernel(*ca, **ck)
    return call


mx.fast.metal_kernel = _logged_kernel
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving import mlx_grad_compat
mlx_grad_compat.apply()                                          # differentiable forward without editing oMLX's files
from drift.serving.omlx_cache import kv_layer_indices

ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire('studio_grad_probe.py', need_gb=175)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
print('plain rotary modules:', mlx_grad_compat.use_plain_rotary(model), flush=True)
passage = "Shipping note. The delivery code is 7291 and the pallet must be signed for by Mr Okafor."
question = "<|im_start|>user\nWhat is the delivery code?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe delivery code is"
p_ids, q_ids, a_ids = (tok.encode(x, add_special_tokens=False).ids for x in (passage, question, " 7291"))
cache = lm.make_cache(); mx.eval(lm(mx.array([p_ids]), cache=cache).logits)
layers = kv_layer_indices(cache); n = len(p_ids)
K = [cache[i].keys[:, :, :n].astype(mx.float32) for i in layers]; V = [cache[i].values[:, :, :n].astype(mx.float32) for i in layers]
IK = [cache[i].index_keys for i in layers]
ids, targets, na = mx.array([q_ids + a_ids]), mx.array(a_ids), len(a_ids)


def loss_fn(dK):
    c = lm.make_cache()
    for j, i in enumerate(layers):
        c[i].update_and_fetch((K[j] + dK[j]).astype(mx.bfloat16), V[j].astype(mx.bfloat16))
        c[i].update_indexer(mx.zeros_like(IK[j]), mx.arange(n, dtype=mx.int32)[None])
    positions = mx.arange(n, n + ids.shape[1], dtype=mx.int32)[None]
    logits = lm(ids, cache=c, position_ids=positions).logits[0, -na - 1:-1].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.take_along_axis(logp, targets[:, None], axis=-1).mean()


zeros = [mx.zeros_like(k) for k in K]
report = {}
for mode in ("eval", "train"):
    getattr(model, mode)()
    KERNEL_CALLS.clear()
    try:
        t = time.time(); val = loss_fn(zeros); mx.eval(val); report[f"{mode}_forward"] = {"loss": float(val), "s": round(time.time() - t, 2), "custom_kernels_called": dict(KERNEL_CALLS)}
        KERNEL_CALLS.clear()
        t = time.time(); val, grads = mx.value_and_grad(loss_fn)(zeros); mx.eval(val, grads)
        report[f"{mode}_grad"] = {"ok": True, "loss": float(val), "grad_norms": [round(float(mx.sqrt((g * g).sum())), 4) for g in grads][:4], "s": round(time.time() - t, 2), "peak_gb": round(mx.get_peak_memory() / 2**30, 1)}
    except Exception as error:
        report[f"{mode}_kernels_during_grad_trace"] = dict(KERNEL_CALLS)
        report[f"{mode}_grad"] = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:300]}", "where": traceback.format_exc().strip().splitlines()[-3][:200]}
print(json.dumps(report, indent=1))
