"""Tap canonical K/V from a local MLX model (mlx-lm) for the offline answer-level training prototype.
Used while the Sparks and the Studio are unreachable: writer Qwen3.6-35B-A3B and reader Qwen3.8-27B (both 4-bit MLX,
same tokenizer, same hybrid design as the Drift reader: full attention every 4th layer, partial RoPE 64/256, theta 1e7).
  .venv-next/bin/python scripts/local/tap_local.py --model <dir> --texts <jsonl|corpus json> --out <dir>"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from drift.serving.omlx_cache import Rope

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--texts", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int)
parser.add_argument("--max-tokens", type=int, default=400)
args = parser.parse_args()
model, tok = load(args.model)
cfg = json.loads((Path(args.model) / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
raw = args.texts.read_text()
texts = [r["text"] for r in json.loads(raw)["records"]] if raw.lstrip().startswith("{\"records\"") or raw.lstrip().startswith("{") and "\"records\"" in raw[:40] else [json.loads(l)["text"] for l in raw.splitlines()]
texts = list(dict.fromkeys(texts))[: args.limit] if args.limit else list(dict.fromkeys(texts))
args.out.mkdir(parents=True, exist_ok=True)
started, done, tokens, checked = time.time(), 0, 0, False
for text in texts:
    pid = hashlib.sha256(text.encode()).hexdigest()[:16]
    target = args.out / f"{pid}.npz"
    if target.exists():
        continue
    ids = tok.encode(text, add_special_tokens=False)[: args.max_tokens]
    cache = model.make_cache()
    mx.eval(model(mx.array([ids]), cache=cache))
    layers = [i for i, c in enumerate(cache) if hasattr(c, "keys") and c.keys is not None]
    n, out = len(ids), {}
    for i in layers:
        k = rope.apply(cache[i].keys[:, :, :n], np.arange(n), inverse=True); v = cache[i].values[:, :, :n].astype(mx.float32)
        if not checked:                                            # my rotation must match THIS runtime's own rotary module, position by position
            inner = getattr(model, "language_model", model); inner = getattr(inner, "model", inner)
            module = inner.layers[i].self_attn.rope
            probe = mx.random.normal((1, 2, 3, int(t["head_dim"])))
            theirs = mx.concatenate([module(probe[:, :, j:j + 1], offset=p) for j, p in enumerate((0, 7, 300))], axis=2).astype(mx.float32)
            err = float(mx.abs(rope.apply(probe, np.array([0, 7, 300])) - theirs).max()); assert err < 1e-3, f"rotary convention mismatch {err}"
            checked = True
        mx.eval(k, v)
        out[f"k{i}"], out[f"v{i}"] = np.array(k)[0].transpose(1, 0, 2).astype(np.float16), np.array(v)[0].transpose(1, 0, 2).astype(np.float16)
    np.savez(target, **out); done += 1; tokens += n
    if done % 100 == 0:
        print(done, "texts", tokens, "tokens", round(time.time() - started), "s", flush=True)
print(json.dumps({"tapped": done, "tokens": tokens, "seconds": round(time.time() - started, 1), "kv_layers": layers}))
