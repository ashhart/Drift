"""Real-weights check of the cache seam inside oMLX's own runtime (Studio). EXPLORATORY, not a
preregistered E1: it measures the runtime's own drift, the tap/inject round trip, identity through
the serving seam, and how much context the KV-bearing layers alone carry (self-transfer upper bound).

Run with oMLX's interpreter:
  R=/Applications/oMLX.app/Contents/Resources
  PYTHONPATH="$R/Python/framework-mlx-base/lib/python3.11/site-packages:$R:." \
    $R/Python/cpython-3.11/bin/python3 scripts/studio_omlx_serving_check.py --out local/studio/omlx_serving_check.json
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, copy_nonkv_state, inject_prefix, kv_layer_indices, tap

ck = Path(args.checkpoint).expanduser()
cfg = json.loads((ck / "config.json").read_text())
text_cfg = cfg.get("text_config", cfg)
rope_params = text_cfg.get("rope_parameters") or text_cfg.get("rope_scaling") or {}
theta = float(rope_params.get("rope_theta", text_cfg.get("rope_theta", 10000.0)))
rotary_dim = int(text_cfg["head_dim"] * float(rope_params.get("partial_rotary_factor", text_cfg.get("partial_rotary_factor", 1.0))))
rope = Rope(theta, rotary_dim)
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
PASSAGES = [
    "The Apollo program was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon from 1969 to 1972. It was first conceived during the Eisenhower administration as a three-person spacecraft to follow the one-person Project Mercury. The first crewed landing took place in July 1969, when Neil Armstrong and Buzz Aldrin landed their lunar module and walked on the surface while Michael Collins remained in orbit.",
    "Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into chemical energy stored in glucose. It takes place mainly in the chloroplasts, where chlorophyll absorbs light. Water and carbon dioxide are the raw materials, and oxygen is released as a by-product. The overall reaction can be summarised as carbon dioxide plus water, in the presence of light, giving glucose and oxygen.",
    "def binary_search(items, target):\n    low, high = 0, len(items) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n\n# binary_search returns the index of target in a sorted list, or -1 when the target is not present.",
]
t0 = time.time()
from drift.serving.studio_guard import acquire
acquire('studio_omlx_serving_check.py', need_gb=150)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
report = {"checkpoint": str(ck), "config_sha256": hashlib.sha256((ck / "config.json").read_bytes()).hexdigest(),
          "runtime": "oMLX bundled python; mlx-vlm 0.6.3 + vendored qwen4_exp", "rope": {"theta": theta, "rotary_dim": rotary_dim},
          "load_seconds": round(time.time() - t0, 1), "label": "EXPLORATORY (not preregistered)", "passages": []}


def logits(ids, cache=None):
    out = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache).logits[0]
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def acc(pred_logits, targets):
    return float((pred_logits.argmax(-1) == np.asarray(targets)).mean())


for text in PASSAGES:
    ids = np.array(tok.encode(text).ids, dtype=np.int32)
    tail = 16
    n = len(ids) - tail
    full = logits(ids)
    native = lm.make_cache()
    logits(ids[:n], native)
    kv_layers = kv_layer_indices(native)
    native_cont = logits(ids[n:], copy.deepcopy(native))
    # 1) tap -> canonical -> re-rotate round trip against the native cache rows
    canonical = tap(native, kv_layers, rope, 0, n)
    probe = lm.make_cache()
    index_keys = {l: native[l].index_keys for l in kv_layers}
    inject_prefix(probe, canonical, rope, index_keys=index_keys)
    roundtrip = max(float(mx.abs(probe[l].keys[:, :, :n].astype(mx.float32) - native[l].keys[:, :, :n].astype(mx.float32)).max()) for l in kv_layers)
    # 2) identity through the seam: KV rebuilt from canonical + recurrent state copied
    copy_nonkv_state(native, probe, kv_layers)
    identity_cont = logits(ids[n:], probe)
    # 3) connector-mode self-transfer: KV prefix only, recurrent state fresh (what a serving connector can do)
    kv_only = lm.make_cache()
    inject_prefix(kv_only, canonical, rope, index_keys=index_keys)
    kv_only_cont = logits(ids[n:], kv_only)
    # 3b) same with zeroed selector keys (what a foreign writer could supply without an indexer head)
    kv_zero = lm.make_cache()
    inject_prefix(kv_zero, canonical, rope, index_keys={l: mx.zeros_like(index_keys[l]) for l in kv_layers})
    kv_zero_cont = logits(ids[n:], kv_zero)
    # 4) floor: no context at all
    floor_cont = logits(ids[n:], lm.make_cache())
    targets = ids[n + 1:]
    row = {
        "tokens": int(len(ids)), "prefix": int(n), "kv_layers": len(kv_layers),
        "runtime_drift_full_vs_cached": {"max_abs": float(np.abs(native_cont - full[n:]).max()), "argmax_agree": float((native_cont.argmax(-1) == full[n:].argmax(-1)).mean())},
        "roundtrip_native_key_max_abs": roundtrip,
        "identity_vs_native": {"max_abs": float(np.abs(identity_cont - native_cont).max()), "argmax_agree": float((identity_cont.argmax(-1) == native_cont.argmax(-1)).mean())},
        "next_token_accuracy_on_tail": {
            "native_full_context": acc(native_cont[:-1], targets),
            "kv_prefix_only_real_selector_keys": acc(kv_only_cont[:-1], targets),
            "kv_prefix_only_zero_selector_keys": acc(kv_zero_cont[:-1], targets),
            "no_context_floor": acc(floor_cont[:-1], targets)},
        "mean_target_logprob_on_tail": {},
    }
    for name, lg in (("native_full_context", native_cont), ("kv_prefix_only_real_selector_keys", kv_only_cont),
                     ("kv_prefix_only_zero_selector_keys", kv_zero_cont), ("no_context_floor", floor_cont)):
        z = lg[:-1] - lg[:-1].max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        row["mean_target_logprob_on_tail"][name] = float(logp[np.arange(len(targets)), targets].mean())
    report["passages"].append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
report["peak_gb"] = round(mx.get_peak_memory() / 2**30, 1)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
