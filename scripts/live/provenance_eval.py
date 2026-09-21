"""Held-out provenance evaluation on the real pair (criteria: configs/preregistration.provenance-heldout.json).
GLM reads passage A on the Sparks; Qwen (Studio drift worker) holds note B as text and gets A only as translated cache
entries. Three drift arms on the same items and memories: the D1b concise arm unchanged (old link prompt, "shared memory"
wording), the source-naming PROTOCOL with the untrained v3 translator, and the protocol with the provenance-trained
correction (loudness gate + source tag). Resumable."""
import argparse, hashlib, json, os, subprocess, sys, time
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_items import build, correct
from livelib import GLM_LAYERS, OMLX_PY, STUDIO, glm_read, load_reader, sh
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--frozen", type=Path, required=True)
parser.add_argument("--prereg", type=Path)
parser.add_argument("--validation", action="store_true")
parser.add_argument("--per-type", type=int, default=40)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if not args.validation and args.prereg is None:
    raise SystemExit("an evaluation run needs --prereg")
STUDIO_DIR = os.environ.get("DRIFT_STUDIO_DIR", "drift-frontier")
frozen = json.loads(args.frozen.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key, path in frozen["artifacts"].items():
    if sha(path) != frozen["sha256"][key]:
        raise SystemExit(f"frozen artifact {key} does not match its recorded sha256")
items, counts = build(args.passages, args.per_type)
if not args.validation and min(counts.values()) < frozen["minimum_items_per_type"]:
    raise SystemExit(f"INVALID: too few items per type {counts}")
print("items", counts, flush=True)
args.out.mkdir(parents=True, exist_ok=True); run, t0 = args.out.name, time.time()
ids = [it["a"]["id"] for it in items]
taps = args.out / "taps_glm"
if len(list(taps.glob("*.npz"))) < len(ids):
    taps = glm_read({it["a"]["id"]: it["a"]["text"] for it in items}, args.out, run)
fan = FanoutReader.load(Path(frozen["artifacts"]["forward_fanout"]), load_reader(Path(frozen["artifacts"]["forward_base"])))
readers = {"untrained": fan, "trained": CorrectedFanoutReader.load(Path(frozen["artifacts"]["provenance_correction"]), fan)}
remote = f"out/runs/{run}"
sh("ssh", STUDIO, f"mkdir -p {STUDIO_DIR}/{remote}/own"); sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:{STUDIO_DIR}/")
for name, reader in readers.items():
    (args.out / name).mkdir(exist_ok=True)
    for pid in ids:
        z = np.load(taps / f"{pid}.npz")
        e = reader.read({l: z[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, frozen["forward_gain_power"])
        np.savez(args.out / name / f"{pid}.npz", **{f"k{l}": k.astype(np.float16) for l, (k, v) in e.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in e.items()})
    sh("rsync", "-a", str(args.out / name), f"{STUDIO}:{STUDIO_DIR}/{remote}/")
partial = args.out / "rows.partial.jsonl"
rows = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
done, worker = {r["id"] for r in rows}, None


def call(**cmd):
    global worker
    if worker is None or worker.poll() is not None:
        worker = subprocess.Popen(["ssh", STUDIO, f"cd ~/{STUDIO_DIR} && {OMLX_PY} scripts/live/studio_drift_worker.py 2>out/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        json.loads(worker.stdout.readline())
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok"):
        raise RuntimeError(reply)
    return reply


def ask(prompt, memory=None, system=None):
    call(op="start")
    if memory:
        call(op="append", memory=memory)
    call(op="extend", chat=prompt, system=system)
    return call(op="generate", tokens=frozen["max_new_tokens"])["text"]


for n, it in enumerate(items):
    pid = it["a"]["id"]
    if pid in done:
        continue
    same = [x for x in items if x["kind"] == it["kind"]]
    other = same[(same.index(it) + 1) % len(same)]
    own = f"{remote}/own/{pid}.npz"
    call(op="start"); call(op="extend", text=it["a"]["text"]); call(op="tap", first=0, out=own)
    old = f"Your own note: {it['note']}\n\nQuestion: {it['question']}{frozen['question_suffix']}"                     # the D1b concise arm, unchanged
    new = f"Your own note: {it['note']}\n\nQuestion: {it['question_protocol']}{frozen['question_suffix']}"            # the source-naming protocol
    P, D = frozen["link_prompt_protocol"], frozen["link_prompt_d1"]
    answers = {"d1b_untrained": ask(old, f"{remote}/untrained/{pid}.npz", D),
               "protocol_untrained": ask(new, f"{remote}/untrained/{pid}.npz", P), "protocol_trained": ask(new, f"{remote}/trained/{pid}.npz", P),
               "wrong_memory": ask(new, f"{remote}/untrained/{other['a']['id']}.npz", P), "own_kv": ask(new, own, P), "no_memory": ask(new, None, P),
               "text_both": ask(f"Document from your partner: {it['a']['text']}\n\n{new}")}
    row = {"id": pid, "kind": it["kind"], "question": it["question"], "question_protocol": it["question_protocol"], "reference": it["reference"], "must": it["must"], "must_not": it["must_not"], **answers,
           "correct": {c: correct(t, it) for c, t in answers.items()}}
    rows.append(row)
    with partial.open("a") as sink:
        sink.write(json.dumps(row) + "\n")
    print(len(rows), it["kind"], {c: int(v) for c, v in row["correct"].items()}, flush=True)
if worker is not None:
    worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush()
conditions = ["d1b_untrained", "protocol_untrained", "protocol_trained", "wrong_memory", "own_kv", "no_memory", "text_both"]
em = lambda rs: {c: round(sum(r["correct"][c] for r in rs) / max(1, len(rs)), 4) for c in conditions}
by_kind, pooled = {k: {"n": sum(r["kind"] == k for r in rows), **em([r for r in rows if r["kind"] == k])} for k in counts}, em(rows)


def sign(rs, better, worse):                                                   # items are independent passage pairs: exact one-sided sign test on discordant items
    wins, losses = sum(r["correct"][better] and not r["correct"][worse] for r in rs), sum(r["correct"][worse] and not r["correct"][better] for r in rs)
    p = sum(comb(wins + losses, k) for k in range(wins, wins + losses + 1)) / 2 ** (wins + losses) if wins + losses else 1.0
    return {"first_only": wins, "second_only": losses, "sign_test_p": p}


conj = [r for r in rows if r["kind"] == "conj"]
tests = {"conj_protocol_vs_d1b": sign(conj, "protocol_untrained", "d1b_untrained"), "pooled_trained_vs_untrained": sign(rows, "protocol_trained", "protocol_untrained")}
t = frozen["thresholds"]
h1 = by_kind["conj"]["protocol_untrained"] - by_kind["conj"]["d1b_untrained"] >= t["H1_conj_gain"] and tests["conj_protocol_vs_d1b"]["sign_test_p"] < 0.05
h2 = by_kind["conj"]["protocol_untrained"] >= t["H2_conj_target"] and by_kind["conj"]["wrong_memory"] <= t["H2_wrong_memory_max"]
h3 = all(by_kind[k]["protocol_untrained"] >= by_kind[k]["d1b_untrained"] - t["H3_no_cost"] for k in ("sum", "hop"))
h4 = pooled["protocol_trained"] - pooled["protocol_untrained"] >= t["H4_training_gain"] and tests["pooled_trained_vs_untrained"]["sign_test_p"] < 0.05
verdict = "PASSED" if h1 and h2 and h3 else "PARTIAL" if h1 and h3 else "FAILED"
report = {"kind": "VALIDATION (pipeline check, not evidence)" if args.validation else "PREREGISTERED EVALUATION (provenance, held out)", "preregistration_sha256": sha(args.prereg) if args.prereg else None,
          "frozen_sha256": sha(args.frozen), "items": len(rows), "pooled": pooled, "by_kind": by_kind, "tests": tests,
          "H1_protocol_fixes_conjunction": h1, "H2_sources_kept_apart": h2, "H3_no_cost_elsewhere": h3, "H4_training_adds": h4,
          "verdict_protocol": None if args.validation else verdict, "verdict_training": None if args.validation else ("ADDS" if h4 else "NO MEASURED GAIN"),
          "seconds": round(time.time() - t0, 1), "rows": rows}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
