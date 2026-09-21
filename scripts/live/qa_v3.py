"""Live QA v3, both directions (criteria: configs/preregistration.live-qa-v3.json; frozen choices: configs/frozen.live-qa-v3.json).
  --direction forward : GLM reads, Qwen answers (Studio drift worker)
  --direction reverse : Qwen reads, GLM answers (vLLM, DriftGlm53Connector)"""
import argparse, hashlib, json, subprocess, sys, time
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizers import Tokenizer
from livelib import GLM_LAYERS, OMLX_PY, QWEN_LAYERS, SPARK, SPARK_PEERS, STUDIO, glm_read, load_reader, passage_id, sh
from questions_v3 import NUMERIC, PRIORITY, QUESTION
from drift.translate.fanout import FanoutReader
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--direction", choices=["forward", "reverse"], required=True)
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--frozen", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int, default=60)
parser.add_argument("--max-new", type=int, default=160)
args = parser.parse_args()
frozen = json.loads(args.frozen.read_text())
glm_tok, qwen_tok = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
items = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line); facts = {f["kind"]: f["value"] for f in r["facts"]}
    kind = next((k for k in PRIORITY[len(items) % len(PRIORITY):] + PRIORITY if k in facts), None)
    if kind and all(t.decode(t.encode(r["text"], add_special_tokens=False).ids) == r["text"] for t in (glm_tok, qwen_tok)):
        items.append({"id": passage_id(r["text"]), "passage": r["text"], "kind": kind, "answer": facts[kind], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"])})
items = items[: args.limit]
args.out.mkdir(parents=True, exist_ok=True); run, t0 = args.out.name, time.time()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key, path in frozen["artifacts"].items():
    if sha(path) != frozen["sha256"][key]:
        raise SystemExit(f"frozen artifact {key} does not match its recorded sha256")
remote = f"local/studio/runs/{run}"
sh("ssh", STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}"); sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")

if args.direction == "forward":
    taps = glm_read({i["id"]: i["passage"] for i in items}, args.out, run)
    base2, base3 = load_reader(Path(frozen["artifacts"]["forward_v2"])), load_reader(Path(frozen["artifacts"]["forward_base"]))
    readers = {"v3": FanoutReader.load(Path(frozen["artifacts"]["forward_fanout"]), base3), "v2": base2}
    for name, reader in readers.items():
        (args.out / name).mkdir(exist_ok=True)
        for it in items:
            z = np.load(taps / f"{it['id']}.npz")
            e = reader.read({l: z[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, frozen["forward_gain_power"])
            np.savez(args.out / name / f"{it['id']}.npz", **{f"k{l}": k.astype(np.float16) for l, (k, v) in e.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in e.items()})
        sh("rsync", "-a", str(args.out / name), f"{STUDIO}:drift/{remote}/")
    sh("ssh", STUDIO, f"mkdir -p drift/{remote}/own")
    worker = subprocess.Popen(["ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_drift_worker.py 2>local/studio/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def call(**cmd):
        worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
        reply = json.loads(worker.stdout.readline())
        if not reply.get("ok", reply.get("ready")):
            raise RuntimeError(reply)
        return reply

    def ask(question, memory=None):
        call(op="start")
        if memory:
            call(op="append", memory=memory)
        call(op="extend", chat=question)
        return call(op="generate", tokens=args.max_new)["text"]

    json.loads(worker.stdout.readline())
    rows = []
    for n, it in enumerate(items):
        other = items[(n + 1) % len(items)]
        own = f"{remote}/own/{it['id']}.npz"
        call(op="start"); call(op="extend", text=it["passage"]); call(op="tap", first=0, out=own)
        rows.append({"no_memory": ask(it["question"]), "drift_v3": ask(it["question"], f"{remote}/v3/{it['id']}.npz"), "drift_v2": ask(it["question"], f"{remote}/v2/{it['id']}.npz"),
                     "wrong_memory": ask(it["question"], f"{remote}/v3/{other['id']}.npz"), "control": ask(it["question"], own), "text_in_prompt": ask(f"{it['passage']}\n\n{it['question']}")})
        print(n + 1, it["kind"], {c: int(it["answer"].lower() in t.lower()) for c, t in rows[-1].items()}, flush=True)
    worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush()
    conditions, control_name = ["no_memory", "drift_v3", "drift_v2", "wrong_memory", "control", "text_in_prompt"], "own_kv"
else:
    (args.out / "qwen.ids.json").write_text(json.dumps({"records": [{"id": i["id"], "ids_qwen": qwen_tok.encode(i["passage"], add_special_tokens=False).ids} for i in items]}))
    sh("scp", "-q", str(args.out / "qwen.ids.json"), f"{STUDIO}:drift/{remote}/ids.json")
    sh("ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_tap_qwen.py --ids {remote}/ids.json --out {remote}/taps 2>/dev/null | tail -1")
    sh("rsync", "-a", f"{STUDIO}:drift/{remote}/taps/", str(args.out / "taps_qwen"))
    readers = {"v3": StackedReader.load(Path(frozen["artifacts"]["reverse_v3"]), QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512),
               "v2": StackedReader.load(Path(frozen["artifacts"]["reverse_v2"]), QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512)}
    own_taps = glm_read({i["id"]: i["passage"] for i in items}, args.out, run)
    (args.out / "inject").mkdir(exist_ok=True)

    def save(latents, name):
        np.savez(args.out / "inject" / f"{name}.npz", **{f"l{l}": latents[l].astype(np.float16) for l in GLM_LAYERS})
        return {"name": name, "rows": int(latents[GLM_LAYERS[0]].shape[0])}

    for it in items:
        z = np.load(args.out / "taps_qwen" / f"{it['id']}.npz"); T = len(z["k3"])
        flat = {l: np.concatenate((z[f"k{l}"].reshape(T, -1), z[f"v{l}"].reshape(T, -1)), axis=1).astype(np.float32) for l in QWEN_LAYERS}
        for name, reader in readers.items():
            it[name] = save(reader.read(flat, frozen["reverse_gain_power"]), f"{run}-{it['id']}-{name}")
        own = np.load(own_taps / f"{it['id']}.npz")
        it["own"] = save({l: own[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, f"{run}-{it['id']}-own")
    jobs = [{"id": it["id"], "question": it["question"], "answer": it["answer"], "passage": it["passage"],
             "memories": {"drift_v3": it["v3"], "drift_v2": it["v2"], "wrong_memory": items[(n + 1) % len(items)]["v3"], "control": it["own"]}} for n, it in enumerate(items)]
    (args.out / "jobs.json").write_text(json.dumps(jobs))
    for host in [*SPARK_PEERS, SPARK]:
        sh("ssh", host, "mkdir -p /dev/shm/glm53-handoff/tp-inject"); sh("rsync", "-a", str(args.out / "inject") + "/", f"{host}:/dev/shm/glm53-handoff/tp-inject/")
    sh("scp", "-q", "scripts/live/spark_run.sh", "scripts/live/spark_inject_answer.py", f"{SPARK}:/root/drift-live/")
    sh("ssh", SPARK, f"mkdir -p /root/drift-live/runs/{run}"); sh("scp", "-q", str(args.out / "jobs.json"), f"{SPARK}:/root/drift-live/runs/{run}/jobs.json")
    # detached on the Spark and polled from here, so a dropped link (or a sleeping laptop) does not kill the run; the job itself resumes
    sh("ssh", SPARK, f"cd /root/drift-live && (nohup ./spark_run.sh spark_inject_answer.py --jobs runs/{run}/jobs.json --out runs/{run}/answers.json --copies {frozen['reverse_copies']} "
                     f"--placeholder-id 198 --max-new {args.max_new} > runs/{run}/job.log 2>&1 < /dev/null; touch runs/{run}/job.done) > /dev/null 2>&1 &")
    while True:
        try:
            if "yes" in sh("ssh", "-o", "ConnectTimeout=15", SPARK, f"test -e /root/drift-live/runs/{run}/job.done && echo yes || echo no"):
                break
        except subprocess.CalledProcessError:
            pass
        time.sleep(30)
    sh("scp", "-q", f"{SPARK}:/root/drift-live/runs/{run}/answers.json", str(args.out / "answers.json"))
    for host in [SPARK, *SPARK_PEERS]:
        sh("ssh", host, f"rm -f /dev/shm/glm53-handoff/tp-inject/{run}-*")
    raw = json.loads((args.out / "answers.json").read_text())["results"]
    conditions, control_name = ["no_memory", "placeholders_only", "drift_v3", "drift_v2", "wrong_memory", "control", "text_in_prompt"], "own_latents"
    rows = [{c: ("" if "error" in r[c].get("inject", {}) else r[c]["text"]) for c in conditions} for r in raw]

hit = lambda it, row, c: it["answer"].lower() in row[c].lower()
em = lambda pairs: {c: round(sum(hit(it, row, c) for it, row in pairs) / max(1, len(pairs)), 3) for c in conditions}
pairs = list(zip(items, rows))
numeric = [(it, row) for it, row in pairs if it["kind"] in NUMERIC]
overall = em(pairs)
wins, losses = sum(hit(i, r, "drift_v3") and not hit(i, r, "wrong_memory") for i, r in pairs), sum(hit(i, r, "wrong_memory") and not hit(i, r, "drift_v3") for i, r in pairs)
p_sign = sum(comb(wins + losses, k) for k in range(wins, wins + losses + 1)) / 2 ** (wins + losses) if wins + losses else 1.0
c1, c2, c3 = (overall["drift_v3"] - overall["wrong_memory"] >= 0.25 and p_sign < 0.05), overall["drift_v3"] >= overall["control"] - 0.05, overall["drift_v3"] >= overall["drift_v2"]
report = {"kind": "PREREGISTERED EVALUATION", "direction": args.direction, "control_is": control_name, "preregistration_sha256": sha(args.prereg), "frozen_sha256": sha(args.frozen), "items": len(items),
          "exact_match": overall, "exact_match_numeric": {"n": len(numeric), **em(numeric)}, "exact_match_other": {"n": len(pairs) - len(numeric), **em([p for p in pairs if p[0]["kind"] not in NUMERIC])},
          "v3_vs_wrong": {"v3_only": wins, "wrong_only": losses, "sign_test_p": p_sign},
          "v3_vs_v2": {"v3_only": sum(hit(i, r, "drift_v3") and not hit(i, r, "drift_v2") for i, r in pairs), "v2_only": sum(hit(i, r, "drift_v2") and not hit(i, r, "drift_v3") for i, r in pairs)},
          "C1_content_specific": c1, "C2_gap_closed": c2, "C3_improved": c3, "verdict": "PASSED" if c1 and c2 and c3 else "PARTIAL" if c1 else "FAILED", "seconds": round(time.time() - t0, 1),
          "rows": [{**{k: it[k] for k in ("id", "kind", "question", "answer")}, **row} for it, row in pairs]}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
