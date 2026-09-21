"""Tap Qwen3.8's RAW sparse-selector keys (unrotated; positions are stored separately by the runtime) for the calibration
corpus. They become a translation target like K and V: beyond the selector budget an entry with a zero selector key is
never chosen (scripts/live/studio_longctx_probe.py: 0% vs 75% at 4k-8k tokens)."""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, kv_layer_indices, tap
from drift.serving.studio_guard import acquire

parser = argparse.ArgumentParser()
parser.add_argument("--ids", type=Path, nargs="+", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--kv", action="store_true", help="also save the canonical K/V (k{layer}, v{layer}) in the same file: one pass for new corpora")
args = parser.parse_args()
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
acquire("studio_tap_qwen_index.py", need_gb=140)
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
args.out.mkdir(parents=True, exist_ok=True)
started, done, tokens = time.time(), 0, 0
for path in args.ids:
    for r in json.loads(path.read_text())["records"]:
        target = args.out / f"{r['id']}.npz"
        if target.exists():
            continue
        ids = np.asarray(r["ids_qwen"], dtype=np.int32)
        cache = lm.make_cache()
        mx.eval(lm(mx.array(ids)[None], cache=cache).logits)
        layers = kv_layer_indices(cache)
        kv = {}
        if args.kv:
            entries = tap(cache, layers, rope, 0, len(ids))
            kv = {**{f"k{l}": k.astype(np.float16) for l, (k, v) in entries.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in entries.items()}}
        np.savez(target, **kv, **{f"i{l}": np.array(cache[l].index_keys[0, :len(ids)].astype(mx.float32)).astype(np.float16) for l in layers})
        done += 1; tokens += len(ids)
print(json.dumps({"tapped": done, "tokens": tokens, "seconds": round(time.time() - started, 1), "index_dim": int(cache[layers[0]].index_keys.shape[-1])}))
