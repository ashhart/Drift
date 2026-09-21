"""D1: reasoning over borrowed memory (criteria: configs/preregistration.d1-split-knowledge.json).
GLM reads passage A on the Sparks; Qwen (Studio drift worker, model stays loaded) holds passage B as text and gets
A only as translated cache entries appended to its running cache."""
import argparse, hashlib, json, random, re, subprocess, sys, time
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizers import Tokenizer
from livelib import OMLX_PY, STUDIO, glm_read, load_reader, passage_id, sh, translate
from drift.translate.fanout import FanoutReader

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--translator", type=Path, default=Path("local/live/stacked2.npz"))
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--per-type", type=int, default=20)
parser.add_argument("--prereg", type=Path)
parser.add_argument("--validation", action="store_true")
parser.add_argument("--max-new", type=int, default=320)
parser.add_argument("--fanout", type=Path, help="fan-out maps on top of --translator (the v3 forward pipeline)")
parser.add_argument("--arm", choices=["original", "concise"], default="original", help="D1b: concise appends a one-sentence instruction and should be run with --max-new 96")
args = parser.parse_args()
if not args.validation and args.prereg is None:
    raise SystemExit("an evaluation run needs --prereg")
LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; you were never shown it as text. "
        "Use what you recall from that shared memory together with what is in this chat. If the shared memory does not contain what is needed, say that it is missing.")
NAMES_F = "Beatrix Cormac Thandiwe Anselm Rosalind Ezekiel Marisol Ulrich Saoirse Lorenzo Ottoline Bartholomew Yasmin Cedric Philippa Desmond Ximena Leopold Winifred Magnus".split()
NAMES_L = "Abernathy Villanueva Oduya Strickland Montague Lefebvre Underhill Calloway Drummond Esposito Fairweather Hargreaves Ingleby Juarez Kettering Lockwood Mwangi Nightingale Pemberton Rutherford".split()
glm_tok, qwen_tok = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
usable = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line)
    facts = {f["kind"]: f["value"] for f in r["facts"]}
    ok = all(k in facts for k in ("code", "person", "weight", "colour")) and all(t.decode(t.encode(r["text"], add_special_tokens=False).ids) == r["text"] for t in (glm_tok, qwen_tok))
    if ok:
        usable.append({**r, "facts": facts, "id": passage_id(r["text"])})
rng, items, queue, counts = random.Random(99), [], list(usable), {"conj": 0, "sum": 0, "hop": 0}
while queue and any(v < args.per_type for v in counts.values()):
    kind = min((k for k in counts if counts[k] < args.per_type), key=lambda k: (counts[k], ["conj", "sum", "hop"].index(k)))
    a = queue.pop(0)
    if kind == "hop":
        others = set()
        while len(others) < 4:
            name = f"{rng.choice(NAMES_F)} {rng.choice(NAMES_L)}"
            if name != a["facts"]["person"]:
                others.add(name)
        people = sorted(others | {a["facts"]["person"]}, key=lambda _: rng.random())
        ext = dict(zip(people, rng.sample(range(200, 990), 5)))
        note = "Staff directory. " + " ".join(f"{p} is on extension {e}." for p, e in ext.items())
        items.append({"kind": kind, "a": a, "note": note, "question": f"What is the phone extension of the person responsible for the {a['thing']}? That person is named in your shared memory; the directory is in your own note.",
                      "must": [str(ext[a["facts"]["person"]])], "must_not": [str(e) for p, e in ext.items() if p != a["facts"]["person"]], "reference": f"{a['facts']['person']} -> {ext[a['facts']['person']]}"})
    else:
        j = next((n for n, b in enumerate(queue) if b["thing"] != a["thing"] and b["facts"]["colour"] != a["facts"]["colour"] and b["facts"]["code"] != a["facts"]["code"]), None)
        if j is None:
            break
        b = queue.pop(j)
        if kind == "conj":
            items.append({"kind": kind, "a": a, "note": b["text"], "question": f"What is the code mentioned in the {a['genre']} about the {a['thing']} (it is in your shared memory), and which colour is mentioned in your own note?",
                          "must": [a["facts"]["code"], b["facts"]["colour"]], "must_not": [b["facts"]["code"], a["facts"]["colour"]], "reference": f"{a['facts']['code']} and {b['facts']['colour']}"})
        else:
            total = int(a["facts"]["weight"]) + int(b["facts"]["weight"])
            items.append({"kind": kind, "a": a, "note": b["text"], "question": f"How many kilograms do the {a['thing']} (described in your shared memory) and the {b['thing']} (described in your own note) weigh together? Work it out and give the total.",
                          "must": [str(total)], "must_not": [], "reference": f"{a['facts']['weight']} + {b['facts']['weight']} = {total}"})
    counts[kind] += 1
if not args.validation and min(counts.values()) < 12:
    raise SystemExit(f"INVALID: too few items per type {counts}")
print("items", counts, flush=True)
args.out.mkdir(parents=True, exist_ok=True)
run, t0 = args.out.name, time.time()
taps = glm_read({it["a"]["id"]: it["a"]["text"] for it in items}, args.out, run)
reader = load_reader(args.translator)
reader = FanoutReader.load(args.fanout, reader) if args.fanout else reader
translate(taps, [it["a"]["id"] for it in items], reader, args.gain_power, args.out)
remote = f"local/studio/runs/{run}"
sh("ssh", STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}/own")
sh("rsync", "-a", str(args.out / "memory"), f"{STUDIO}:drift/{remote}/")
sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
worker = subprocess.Popen(["ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_drift_worker.py 2>local/studio/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def call(**cmd):
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok", reply.get("ready")):
        raise RuntimeError(reply)
    return reply


def ask(prompt, memory=None, system=LINK):
    call(op="start")
    if memory:
        call(op="append", memory=memory)
    call(op="extend", chat=prompt, system=system)
    return call(op="generate", tokens=args.max_new)["text"]


def correct(text, it):
    has = lambda v: re.search(rf"(?<![\w]){re.escape(v.lower())}(?![\w])", text.lower().replace(",", "")) is not None
    return all(has(v) for v in it["must"]) and not any(has(v) for v in it["must_not"])


json.loads(worker.stdout.readline())
rows = []
for n, it in enumerate(items):
    same = [x for x in items if x["kind"] == it["kind"]]
    other = same[(same.index(it) + 1) % len(same)]
    own_path = f"{remote}/own/{it['a']['id']}.npz"
    call(op="start"); call(op="extend", text=it["a"]["text"]); call(op="tap", first=0, out=own_path)
    prompt = f"Your own note: {it['note']}\n\nQuestion: {it['question']}" + (" Answer in one short sentence, giving the value(s) directly." if args.arm == "concise" else "")
    answers = {"drift": ask(prompt, f"{remote}/memory/{it['a']['id']}.npz"), "wrong_memory": ask(prompt, f"{remote}/memory/{other['a']['id']}.npz"), "no_memory": ask(prompt),
               "own_kv": ask(prompt, own_path), "text_both": ask(f"Document from your partner: {it['a']['text']}\n\n{prompt}", system=None)}
    rows.append({"kind": it["kind"], "question": it["question"], "reference": it["reference"], "note": it["note"], "passage_a": it["a"]["text"],
                 **answers, "correct": {c: correct(t, it) for c, t in answers.items()}})
    print(n + 1, it["kind"], {c: int(v) for c, v in rows[-1]["correct"].items()}, flush=True)
worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush()
conditions = ["drift", "wrong_memory", "no_memory", "own_kv", "text_both"]
em = lambda rs: {c: round(sum(r["correct"][c] for r in rs) / max(1, len(rs)), 3) for c in conditions}
by_kind = {k: {"n": sum(r["kind"] == k for r in rows), **em([r for r in rows if r["kind"] == k])} for k in counts}
pooled = em(rows)
wins, losses = sum(r["correct"]["drift"] and not r["correct"]["wrong_memory"] for r in rows), sum(r["correct"]["wrong_memory"] and not r["correct"]["drift"] for r in rows)
p_sign = sum(comb(wins + losses, k) for k in range(wins, wins + losses + 1)) / 2 ** (wins + losses) if wins + losses else 1.0
c1 = pooled["drift"] - pooled["wrong_memory"] >= 0.25 and p_sign < 0.05
c2 = all(by_kind[k]["drift"] - by_kind[k]["wrong_memory"] >= 0.25 for k in ("sum", "hop"))
c3 = pooled["drift"] >= 0.5 * pooled["own_kv"]
report = {"kind": "VALIDATION (pipeline check, not evidence)" if args.validation else "PREREGISTERED EVALUATION", "preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest() if args.prereg else None,
          "items": len(rows), "pooled": pooled, "by_kind": by_kind, "discordant": {"drift_only": wins, "wrong_only": losses, "sign_test_p": p_sign},
          "C1_flows": c1, "C2_reasons": c2, "C3_useful": c3, "verdict": None if args.validation else ("PASSED" if c1 and c2 and c3 else "PARTIAL" if c1 else "FAILED"),
          "link_prompt": LINK, "arm": args.arm, "seconds": round(time.time() - t0, 1), "rows": rows}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "link_prompt")}, indent=1))
