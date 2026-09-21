"""Test A evaluation (criteria: configs/preregistration.v4-answer-level.json; frozen choices: configs/frozen.v4-answer-level.json).
GLM reads each passage; Qwen (Studio drift worker) answers TWO questions per passage under six conditions. v4 vs v3 is
decided on passage-level differences with the exact clustered sign-flip test and a passage bootstrap. Resumable."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import GLM_LAYERS, OMLX_PY, STUDIO, glm_read, load_reader, passage_id, sh
from questions_v3 import NUMERIC, PRIORITY, QUESTION
from drift.eval.paired_stats import clustered_bootstrap, passage_differences, sign_flip_p
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--frozen", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int, default=120)
parser.add_argument("--max-new", type=int, default=160)
args = parser.parse_args()
frozen = json.loads(args.frozen.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key, path in frozen["artifacts"].items():
    if sha(path) != frozen["sha256"][key]:
        raise SystemExit(f"frozen artifact {key} does not match its recorded sha256")
passages = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line); facts = {f["kind"]: f["value"] for f in r["facts"]}
    start = len(passages) % len(PRIORITY)
    kinds = [k for k in PRIORITY[start:] + PRIORITY[:start] if k in facts][:2]          # two different fact kinds, rotating start
    if len(kinds) == 2:
        passages.append({"id": passage_id(r["text"]), "text": r["text"], "questions": [{"kind": k, "answer": facts[k], "question": QUESTION[k].format(genre=r["genre"], thing=r["thing"])} for k in kinds]})
passages = passages[: args.limit]
args.out.mkdir(parents=True, exist_ok=True); run, t0 = args.out.name, time.time()
taps = glm_read({p["id"]: p["text"] for p in passages}, args.out, run) if not (args.out / "taps_glm").exists() or len(list((args.out / "taps_glm").glob("*.npz"))) < len(passages) else args.out / "taps_glm"
fan = FanoutReader.load(Path(frozen["artifacts"]["forward_fanout"]), load_reader(Path(frozen["artifacts"]["forward_base"])))
readers = {"v3": fan, "v4": CorrectedFanoutReader.load(Path(frozen["artifacts"]["correction"]), fan)}
remote = f"local/studio/runs/{run}"
sh("ssh", STUDIO, f"mkdir -p drift/{remote}/own"); sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
for name, reader in readers.items():
    (args.out / name).mkdir(exist_ok=True)
    for p in passages:
        z = np.load(taps / f"{p['id']}.npz")
        e = reader.read({l: z[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, frozen["forward_gain_power"])
        np.savez(args.out / name / f"{p['id']}.npz", **{f"k{l}": k.astype(np.float16) for l, (k, v) in e.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in e.items()})
    sh("rsync", "-a", str(args.out / name), f"{STUDIO}:drift/{remote}/")
partial = args.out / "rows.partial.jsonl"
rows = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
done = {(r["passage"], r["kind"]) for r in rows}
worker = None


def call(**cmd):
    global worker
    if worker is None or worker.poll() is not None:
        worker = subprocess.Popen(["ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_drift_worker.py 2>local/studio/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        json.loads(worker.stdout.readline())
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok"):
        raise RuntimeError(reply)
    return reply


def ask(question, memory=None):
    call(op="start")
    if memory:
        call(op="append", memory=memory)
    call(op="extend", chat=question)
    return call(op="generate", tokens=args.max_new)["text"]


for n, p in enumerate(passages):
    other = passages[(n + 1) % len(passages)]
    todo = [q for q in p["questions"] if (p["id"], q["kind"]) not in done]
    if not todo:
        continue
    own = f"{remote}/own/{p['id']}.npz"
    call(op="start"); call(op="extend", text=p["text"]); call(op="tap", first=0, out=own)
    for q in todo:
        answers = {"no_memory": ask(q["question"]), "drift_v3": ask(q["question"], f"{remote}/v3/{p['id']}.npz"), "drift_v4": ask(q["question"], f"{remote}/v4/{p['id']}.npz"),
                   "wrong_memory": ask(q["question"], f"{remote}/v4/{other['id']}.npz"), "own_kv": ask(q["question"], own), "text_in_prompt": ask(f"{p['text']}\n\n{q['question']}")}
        row = {"passage": p["id"], **q, **answers}
        rows.append(row)
        with partial.open("a") as sink:
            sink.write(json.dumps(row) + "\n")
        print(len(rows), q["kind"], {c: int(q["answer"].lower() in t.lower()) for c, t in answers.items()}, flush=True)
worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush()
conditions = ["no_memory", "drift_v3", "drift_v4", "wrong_memory", "own_kv", "text_in_prompt"]
hit = lambda r, c: r["answer"].lower() in r[c].lower()
em = lambda rs: {c: round(sum(hit(r, c) for r in rs) / max(1, len(rs)), 4) for c in conditions}
overall = em(rows)
outcomes = {}
for r in rows:
    outcomes.setdefault(r["passage"], []).append((hit(r, "drift_v4"), hit(r, "drift_v3")))
differences = passage_differences(outcomes)
p_value, interval = sign_flip_p(list(differences.values())), clustered_bootstrap(outcomes)
gain = overall["drift_v4"] - overall["drift_v3"]
c0, c1, c3 = overall["drift_v4"] >= 0.95, overall["drift_v4"] >= overall["own_kv"] - 0.05, overall["wrong_memory"] <= 0.10
c2 = gain >= 0.04 and p_value < 0.05
verdict = "PASSED" if c0 and c1 and c2 and c3 else "PARTIAL" if c2 and c3 else "INCONCLUSIVE" if gain >= 0.04 and c3 else "FAILED"
report = {"kind": "PREREGISTERED EVALUATION (Test A)", "preregistration_sha256": sha(args.prereg), "frozen_sha256": sha(args.frozen), "passages": len(outcomes), "questions": len(rows),
          "exact_match": overall, "exact_match_numeric": em([r for r in rows if r["kind"] in NUMERIC]), "exact_match_other": em([r for r in rows if r["kind"] not in NUMERIC]),
          "v4_minus_v3": {"estimate": round(gain, 4), "passage_bootstrap_95": [round(interval["low"], 4), round(interval["high"], 4)], "passages_v4_better": sum(d > 0 for d in differences.values()),
                          "passages_v3_better": sum(d < 0 for d in differences.values()), "clustered_sign_flip_p": p_value},
          "C0_target_0.95": c0, "C1_gap_closed": c1, "C2_improved": c2, "C3_still_specific": c3, "verdict": verdict, "seconds": round(time.time() - t0, 1), "rows": rows}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
