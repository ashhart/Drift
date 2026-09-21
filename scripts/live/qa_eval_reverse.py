"""QA evaluation of the live Qwen -> GLM channel (criteria: configs/preregistration.live-qa-v2-reverse.json)."""
import argparse, hashlib, json, sys, time
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizers import Tokenizer
from livelib import load_packs, GLM_LAYERS, OMLX_PY, QWEN_LAYERS, SPARK, SPARK_PEERS, STUDIO, glm_read, passage_id, sh
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--translator", type=Path)
parser.add_argument("--packs", type=Path)
parser.add_argument("--gain-power", type=float, required=True)
parser.add_argument("--copies", type=float, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--prereg", type=Path)
parser.add_argument("--validation", action="store_true")
parser.add_argument("--limit", type=int)
parser.add_argument("--max-new", type=int, default=160)
parser.add_argument("--placeholder-id", type=int, default=198)
args = parser.parse_args()
if not args.validation and args.prereg is None:
    raise SystemExit("an evaluation run needs --prereg")
src = (Path(__file__).resolve().parent / "qa_eval.py").read_text()
exec(src[src.index("PRIORITY = ["):src.index("glm, qwen = Tokenizer")])                       # the same fixed question templates as the forward evaluation
glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
items = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line)
    facts = {f["kind"]: f["value"] for f in r["facts"]}
    kind = next((k for k in PRIORITY[len(items) % len(PRIORITY):] + PRIORITY if k in facts), None)
    text = r["text"]
    if kind is None or glm.decode(glm.encode(text, add_special_tokens=False).ids) != text or qwen.decode(qwen.encode(text, add_special_tokens=False).ids) != text:
        continue
    items.append({"id": passage_id(text), "passage": text, "kind": kind, "answer": facts[kind], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"])})
items = items[: args.limit] if args.limit else items
args.out.mkdir(parents=True, exist_ok=True)
run, t0 = args.out.name, time.time()
(args.out / "qwen.ids.json").write_text(json.dumps({"records": [{"id": i["id"], "ids_qwen": qwen.encode(i["passage"], add_special_tokens=False).ids} for i in items]}))
remote = f"local/studio/runs/{run}"
sh("ssh", STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}")
sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
sh("scp", "-q", str(args.out / "qwen.ids.json"), f"{STUDIO}:drift/{remote}/ids.json")
print("Qwen:", sh("ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_tap_qwen.py --ids {remote}/ids.json --out {remote}/taps 2>/dev/null | tail -1").strip())
sh("rsync", "-a", f"{STUDIO}:drift/{remote}/taps/", str(args.out / "taps_qwen"))
reader = StackedReader.load(args.translator, QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512) if args.translator else None
packs = load_packs(args.packs) if args.packs else None
(args.out / "inject").mkdir(exist_ok=True)


def save(latents, name):
    np.savez(args.out / "inject" / f"{name}.npz", **{f"l{l}": latents[l].astype(np.float16) for l in GLM_LAYERS})
    return {"name": name, "rows": int(latents[GLM_LAYERS[0]].shape[0])}


own_taps = glm_read({i["id"]: i["passage"] for i in items}, args.out, run)
for it in items:
    z = np.load(args.out / "taps_qwen" / f"{it['id']}.npz")
    flat = {l: np.concatenate((z[f"k{l}"].reshape(len(z[f"k{l}"]), -1), z[f"v{l}"].reshape(len(z[f"v{l}"]), -1)), axis=1).astype(np.float32) for l in QWEN_LAYERS}
    if packs:                                                         # Qwen entries -> pool.v2 rows -> GLM latents
        latents = packs[0].read(packs[1].write({l: (z[f"k{l}"], z[f"v{l}"]) for l in QWEN_LAYERS}), packs[1].pool_fingerprint, args.gain_power)
    else:
        latents = reader.read(flat, args.gain_power)
    it["tp"] = save(latents, f"{run}-{it['id']}-tp")
    own = np.load(own_taps / f"{it['id']}.npz")
    it["own"] = save({l: own[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, f"{run}-{it['id']}-own")
jobs = [{"id": it["id"], "question": it["question"], "answer": it["answer"], "passage": it["passage"],
         "memories": {"drift": it["tp"], "wrong_memory": items[(n + 1) % len(items)]["tp"], "own_latents": it["own"]}} for n, it in enumerate(items)]
(args.out / "jobs.json").write_text(json.dumps(jobs))
for host in [SPARK, *SPARK_PEERS]:
    sh("ssh", host, "mkdir -p /dev/shm/glm53-handoff/tp-inject")
    sh("rsync", "-a", str(args.out / "inject") + "/", f"{host}:/dev/shm/glm53-handoff/tp-inject/")
sh("scp", "-q", "scripts/live/spark_run.sh", "scripts/live/spark_inject_answer.py", f"{SPARK}:/root/drift-live/")
sh("ssh", SPARK, f"mkdir -p /root/drift-live/runs/{run}")
sh("scp", "-q", str(args.out / "jobs.json"), f"{SPARK}:/root/drift-live/runs/{run}/jobs.json")
print("GLM:", sh("ssh", SPARK, f"/root/drift-live/spark_run.sh spark_inject_answer.py --jobs runs/{run}/jobs.json --out runs/{run}/answers.json --copies {args.copies} "
                 f"--placeholder-id {args.placeholder_id} --max-new {args.max_new} | tail -1").strip())
sh("scp", "-q", f"{SPARK}:/root/drift-live/runs/{run}/answers.json", str(args.out / "answers.json"))
for host in [SPARK, *SPARK_PEERS]:
    sh("ssh", host, f"rm -f /dev/shm/glm53-handoff/tp-inject/{run}-*")
rows = json.loads((args.out / "answers.json").read_text())["results"]
conditions = ["no_memory", "placeholders_only", "drift", "wrong_memory", "own_latents", "text_in_prompt"]
written = lambda r, c: "inject" not in r[c] or "error" not in r[c]["inject"]
hit = lambda r, c: written(r, c) and r["answer"].lower() in r[c]["text"].lower()
em = {c: sum(hit(r, c) for r in rows) / len(rows) for c in conditions}
failures = {c: sum(not written(r, c) for r in rows) for c in ("drift", "wrong_memory", "own_latents")}
wins, losses = sum(hit(r, "drift") and not hit(r, "wrong_memory") for r in rows), sum(hit(r, "wrong_memory") and not hit(r, "drift") for r in rows)
p_sign = sum(comb(wins + losses, k) for k in range(wins, wins + losses + 1)) / 2 ** (wins + losses) if wins + losses else 1.0
by_kind = {}
for it, r in zip(items, rows):
    k = by_kind.setdefault(it["kind"], {"n": 0, "drift": 0, "own_latents": 0}); k["n"] += 1; k["drift"] += hit(r, "drift"); k["own_latents"] += hit(r, "own_latents")
c1, c2 = (em["drift"] - em["wrong_memory"] >= 0.25 and p_sign < 0.05), em["drift"] >= 0.5 * em["own_latents"]
report = {"kind": "VALIDATION (hyperparameter selection, not evidence)" if args.validation else "PREREGISTERED EVALUATION", "direction": "qwen -> glm",
          "preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest() if args.prereg else None, "translator_sha256": [x.sha256 for x in packs] if packs else reader.sha256, "translator_meta": packs[0].meta if packs else reader.meta,
          "gain_power": args.gain_power, "copies": args.copies, "items": len(rows), "exact_match": em, "write_failures": failures,
          "discordant": {"drift_only": wins, "wrong_only": losses, "sign_test_p": p_sign}, "by_kind": by_kind, "C1_content_specific": c1, "C2_useful": c2,
          "verdict": None if args.validation else ("PASSED" if c1 and c2 else "PARTIAL" if c1 else "FAILED"), "seconds": round(time.time() - t0, 1),
          "rows": [{**{k: it[k] for k in ("id", "kind", "question", "answer")}, **{c: r[c]["text"] for c in conditions}, "layout": r["drift"].get("layout")} for it, r in zip(items, rows)]}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "translator_meta", "by_kind")}, indent=1))
