"""Reader-side evaluation on the Studio (oMLX runtime): does GLM-written memory help Qwen predict its tail?
Conditions and criteria: configs/preregistration.live-glm-to-qwen.json."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, inject_prefix, kv_layer_indices, tap

parser = argparse.ArgumentParser()
parser.add_argument("--ids", type=Path, required=True)
parser.add_argument("--inject", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--checkpoint", default="~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
args = parser.parse_args()
ck = Path(args.checkpoint).expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
from drift.serving.studio_guard import acquire
acquire('studio_eval_inject.py', need_gb=150)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
records = {r["id"]: r for r in json.loads(args.ids.read_text())["records"]}
plan = json.loads((args.inject / "plan.json").read_text())


def run(ids, cache):
    out = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache).logits[0]
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def score(logits, targets):
    z = logits[:-1] - logits[:-1].max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return float(logp[np.arange(len(targets)), targets].mean()), float((logits[:-1].argmax(-1) == targets).mean())


def injected(entries):
    cache = lm.make_cache()
    layers = kv_layer_indices(cache)
    n = next(iter(entries.values()))[0].shape[0]
    inject_prefix(cache, entries, rope, dtype=mx.bfloat16, index_keys={l: mx.zeros((1, n, INDEX_DIM), dtype=mx.bfloat16) for l in layers})
    return cache


# selector key width, read once from a real prefill
probe = lm.make_cache(); run([1, 2, 3, 4], probe)
INDEX_DIM = int(probe[kv_layer_indices(probe)[0]].index_keys.shape[-1])
load = lambda path, m=None: {int(k[1:]): (z[k][:m].astype(np.float32), z["v" + k[1:]][:m].astype(np.float32)) for z in [np.load(path)] for k in z.files if k.startswith("k")}
rows, started = [], time.time()
for i, item in enumerate(plan):
    r = records[item["id"]]
    ids, n = np.asarray(r["ids_qwen"], dtype=np.int32), item["prefix"]
    tail, targets = ids[n:], ids[n + 1:]
    native = lm.make_cache(); run(ids[:n], native)
    own = tap(native, kv_layer_indices(native), rope, 0, n)
    other = plan[(i + 1) % len(plan)]
    conditions = {"native": native, "self_kv": injected(own), "drift": injected(load(args.inject / f"{item['id']}.npz")),
                  "wrong": injected(load(args.inject / f"{other['id']}.npz", item["entries"])), "floor": lm.make_cache()}
    row = {"id": item["id"], "source": r.get("source"), "prefix": int(n), "entries": item["entries"]}
    for name, cache in conditions.items():
        row[name], row[name + "_acc"] = score(run(tail, cache), targets)
    rows.append(row)
names = ["native", "self_kv", "drift", "wrong", "floor"]
mean = {k: float(np.mean([r[k] for r in rows])) for k in names}
wins_wrong = sum(r["drift"] > r["wrong"] for r in rows); wins_floor = sum(r["drift"] > r["floor"] for r in rows)
c1, c2 = wins_wrong >= 17, (mean["drift"] > mean["floor"] and wins_floor >= 17)
gap = mean["self_kv"] - mean["floor"]
report = {"preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest(), "chunks": len(rows), "mean_logprob": mean,
          "mean_top1": {k: float(np.mean([r[k + "_acc"] for r in rows])) for k in names},
          "drift_beats_wrong": wins_wrong, "drift_beats_floor": wins_floor,
          "fraction_of_self_kv_gap_closed": (mean["drift"] - mean["floor"]) / gap if gap > 0 else None,
          "C1_content_specific": c1, "C2_useful": c2, "verdict": "PASSED" if c1 and c2 else "PARTIAL" if c1 else "FAILED",
          "seconds": round(time.time() - started, 1), "rows": rows}
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
