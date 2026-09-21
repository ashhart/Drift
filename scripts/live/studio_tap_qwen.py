"""Tap Qwen3.8 canonical K/V through the oMLX cache seam. Run with oMLX's bundled interpreter."""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, kv_layer_indices, tap

parser = argparse.ArgumentParser()
parser.add_argument("--ids", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
args = parser.parse_args()
ck = Path(args.checkpoint).expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
from drift.serving.studio_guard import acquire
acquire('studio_tap_qwen.py', need_gb=140)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
args.out.mkdir(parents=True, exist_ok=True)
started, done, tokens = time.time(), 0, 0
for r in json.loads(args.ids.read_text())["records"]:
    target = args.out / f"{r['id']}.npz"
    if target.exists():
        continue
    ids = np.asarray(r["ids_qwen"], dtype=np.int32)
    cache = lm.make_cache()
    mx.eval(lm(mx.array(ids)[None], cache=cache).logits)
    layers = kv_layer_indices(cache)
    entries = tap(cache, layers, rope, 0, len(ids))
    np.savez(target, **{f"k{l}": k.astype(np.float16) for l, (k, v) in entries.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in entries.items()})
    done += 1; tokens += len(ids)
print(json.dumps({"tapped": done, "tokens": tokens, "seconds": round(time.time() - started, 1), "kv_layers": layers}))
