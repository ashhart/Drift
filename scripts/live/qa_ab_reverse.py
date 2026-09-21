"""Fast A/B of reverse-direction (Qwen -> GLM) translator variants on validation passages. Not evidence."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import GLM_LAYERS, OMLX_PY, QWEN_LAYERS, SPARK, SPARK_PEERS, STUDIO, passage_id, sh
from tokenizers import Tokenizer
from drift.translate.merge import MergeFilter, MergeReader
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--translator", type=Path, default=Path("local/live/stacked2_rev.npz"))
parser.add_argument("--merge", type=Path, required=True)
parser.add_argument("--residual", type=Path)
parser.add_argument("--alt-translator", type=Path, help="a second base translator to compare (all rows kept)")
parser.add_argument("--kinds", default="code,weight,count,room,time")
parser.add_argument("--limit", type=int, default=50)
parser.add_argument("--gain-power", type=float, default=1.0)
parser.add_argument("--copies", type=float, default=12)
parser.add_argument("--max-new", type=int, default=160)
parser.add_argument("--system", default="")
args = parser.parse_args()
src = (Path(__file__).resolve().parent / "qa_eval.py").read_text()
exec(src[src.index("PRIORITY = ["):src.index("glm, qwen = Tokenizer")])
qwen = Tokenizer.from_file("local/tok/qwen/tokenizer.json")
items, texts = [], {}
for line in args.passages.read_text().splitlines():
    r = json.loads(line); facts = {f["kind"]: f["value"] for f in r["facts"]}
    for kind in [k for k in args.kinds.split(",") if k in facts]:
        items.append({"id": passage_id(r["text"]), "kind": kind, "answer": facts[kind], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"])}); texts[items[-1]["id"]] = r["text"]
items = items[: args.limit]
args.out.mkdir(parents=True, exist_ok=True); run = args.out.name; remote = f"local/studio/runs/{run}"
ids = sorted({i["id"] for i in items})
(args.out / "qwen.ids.json").write_text(json.dumps({"records": [{"id": pid, "ids_qwen": qwen.encode(texts[pid], add_special_tokens=False).ids} for pid in ids]}))
sh("ssh", STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}"); sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
sh("scp", "-q", str(args.out / "qwen.ids.json"), f"{STUDIO}:drift/{remote}/ids.json")
sh("ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_tap_qwen.py --ids {remote}/ids.json --out {remote}/taps 2>/dev/null | tail -1")
sh("rsync", "-a", f"{STUDIO}:drift/{remote}/taps/", str(args.out / "taps_qwen"))
reader, merge = StackedReader.load(args.translator, QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512), MergeFilter.load(args.merge)
(args.out / "inject").mkdir(exist_ok=True)
specs, dropped = {}, 0
for pid in ids:
    z = np.load(args.out / "taps_qwen" / f"{pid}.npz"); T = len(z["k3"])
    flat = {l: np.concatenate((z[f"k{l}"].reshape(T, -1), z[f"v{l}"].reshape(T, -1)), axis=1).astype(np.float32) for l in QWEN_LAYERS}
    latents = reader.read(flat, args.gain_power)
    keep = merge.keep(np.concatenate([flat[l] for l in merge.meta["layers"]], axis=1)); dropped += int((~keep).sum())
    variants = {"base": latents}
    if args.residual:
        mr = MergeReader.load(args.residual, reader, merge)
        variants["merge_res"] = mr.read(flat, args.gain_power, drop=True); variants["res_keep_all"] = mr.read(flat, args.gain_power, drop=False)
    else:
        variants["merge"] = {l: latents[l][keep] for l in GLM_LAYERS}
    if args.alt_translator:
        variants["alt_base"] = StackedReader.load(args.alt_translator, QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512).read(flat, args.gain_power)
    for name, lat in variants.items():
        np.savez(args.out / "inject" / f"{run}-{pid}-{name}.npz", **{f"l{l}": lat[l].astype(np.float16) for l in GLM_LAYERS})
        specs[(pid, name)] = {"name": f"{run}-{pid}-{name}", "rows": int(lat[GLM_LAYERS[0]].shape[0])}
    NAMES = list(variants)
print("entries dropped by the merge filter:", dropped, flush=True)
jobs = [{"id": f"{it['id']}-{n}", "question": it["question"], "answer": it["answer"], "memories": {name: specs[(it["id"], name)] for name in NAMES}} for n, it in enumerate(items)]
(args.out / "jobs.json").write_text(json.dumps(jobs))
for host in [*SPARK_PEERS, SPARK]:
    sh("ssh", host, "mkdir -p /dev/shm/glm53-handoff/tp-inject"); sh("rsync", "-a", str(args.out / "inject") + "/", f"{host}:/dev/shm/glm53-handoff/tp-inject/")
sh("scp", "-q", "scripts/live/spark_run.sh", "scripts/live/spark_inject_answer.py", f"{SPARK}:/root/drift-live/")
sh("ssh", SPARK, f"mkdir -p /root/drift-live/runs/{run}"); sh("scp", "-q", str(args.out / "jobs.json"), f"{SPARK}:/root/drift-live/runs/{run}/jobs.json")
sh("ssh", SPARK, f"/root/drift-live/spark_run.sh spark_inject_answer.py --jobs runs/{run}/jobs.json --out runs/{run}/answers.json --copies {args.copies} --placeholder-id 198 --max-new {args.max_new} --skip-controls --system {json.dumps(args.system)} | tail -1")
sh("scp", "-q", f"{SPARK}:/root/drift-live/runs/{run}/answers.json", str(args.out / "answers.json"))
for host in [SPARK, *SPARK_PEERS]:
    sh("ssh", host, f"rm -f /dev/shm/glm53-handoff/tp-inject/{run}-*")
rows = json.loads((args.out / "answers.json").read_text())["results"]
for name in NAMES:
    ok = [r["answer"].lower() in r[name]["text"].lower() and "error" not in r[name].get("inject", {}) for r in rows]
    print(name, round(sum(ok) / len(rows), 3), "misses:", [r["answer"] for r, o in zip(rows, ok) if not o])
