"""Sidecar step on the orchestrating host: GLM latents -> pool.v1 -> Qwen canonical K/V (torch).
Writes one npz per record with k<layer>/v<layer> float16 [M,2,256] for all 12 Qwen KV layers."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from drift.translate.pool import PoolFormat, load_translator

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, default=Path("local/live/corpus.json"))
parser.add_argument("--glm", type=Path, default=Path("local/live/taps_glm"))
parser.add_argument("--translators", type=Path, default=Path("local/live/translators"))
parser.add_argument("--out", type=Path, default=Path("local/live/inject_qwen"))
parser.add_argument("--tail", type=int, default=16)
parser.add_argument("--split", default="heldout")
args = parser.parse_args()
p = json.loads((args.translators / "pool.json").read_text())
fmt = PoolFormat("pool.v1", p["levels"], p["width"], p["fingerprint"])
glm, qwen, tail = (load_translator(args.translators / n, fmt) for n in ("glm", "qwen", "qwen_tail"))
args.out.mkdir(parents=True, exist_ok=True)
plan = []
for r in json.loads(args.corpus.read_text())["records"]:
    if r["split"] != args.split:
        continue
    prefix = len(r["ids_qwen"]) - args.tail
    g_rows = [g for g, q in r["aligned"] if q < prefix]
    taps = np.load(args.glm / f"{r['id']}.npz")
    with torch.no_grad():
        pool = glm.write({layer: torch.from_numpy(taps[f"l{layer}"][g_rows].astype(np.float32)) for layer in glm.level_map})
        entries = {**qwen.read(pool), **tail.read(pool)}
    np.savez(args.out / f"{r['id']}.npz", **{f"k{l}": e.k.numpy().astype(np.float16) for l, e in entries.items()},
             **{f"v{l}": e.v.numpy().astype(np.float16) for l, e in entries.items()})
    plan.append({"id": r["id"], "prefix": prefix, "entries": len(g_rows)})
(args.out / "plan.json").write_text(json.dumps(plan))
print(json.dumps({"records": len(plan), "entries": sum(x["entries"] for x in plan)}))
