"""QA evaluation of the live GLM -> Qwen channel (criteria: configs/preregistration.live-qa-v2.json).

  PYTHONPATH=. .venv/bin/python scripts/live/qa_eval.py --passages local/live/gen_eval.jsonl --translator local/live/stacked2.npz \
      --gain-power 1.0 --out local/live/qa_v2 --prereg configs/preregistration.live-qa-v2.json
With --validation it only reports (used for choosing ridge / gain power on --seed 11 passages)."""
import argparse, hashlib, json, sys, time
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizers import Tokenizer
from livelib import glm_read, load_packs, load_reader, passage_id, qwen_answer, translate, translate_packs

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--translator", type=Path)
parser.add_argument("--packs", type=Path, help="pool.v2 model packs directory (glm.pack.npz, qwen.pack.npz) instead of a direct translator")
parser.add_argument("--gain-power", type=float, nargs="+", default=[1.0])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--prereg", type=Path)
parser.add_argument("--validation", action="store_true")
parser.add_argument("--limit", type=int)
parser.add_argument("--max-new", type=int, default=64)
args = parser.parse_args()
if not args.validation and (args.prereg is None or len(args.gain_power) != 1):
    raise SystemExit("an evaluation run needs --prereg and exactly one frozen --gain-power")
PRIORITY = ["code", "person", "time", "ingredient", "animal", "colour", "count", "weight", "room", "day"]
QUESTION = {"code": "What is the code mentioned in the {genre} about the {thing}?", "person": "Who is the person responsible in the {genre} about the {thing}?",
            "time": "What time is given in the {genre} about the {thing}?", "ingredient": "What is the special ingredient mentioned in the {genre} about the {thing}?",
            "animal": "Which animal is involved in the {genre} about the {thing}?", "colour": "Which colour is mentioned in the {genre} about the {thing}?",
            "count": "How many items are mentioned in the {genre} about the {thing}?", "weight": "What weight in kilograms is given in the {genre} about the {thing}?",
            "room": "What is the room or plot number in the {genre} about the {thing}?", "day": "Which day of the week is mentioned in the {genre} about the {thing}?"}
glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
items = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line)
    facts = {f["kind"]: f["value"] for f in r["facts"]}
    kind = next((k for k in PRIORITY[len(items) % len(PRIORITY):] + PRIORITY if k in facts), None)      # rotate the starting kind so every kind is asked
    text = r["text"]
    if kind is None or glm.decode(glm.encode(text, add_special_tokens=False).ids) != text or qwen.decode(qwen.encode(text, add_special_tokens=False).ids) != text:
        continue
    items.append({"id": passage_id(text), "passage": text, "kind": kind, "answer": facts[kind], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"])})
items = items[: args.limit] if args.limit else items
args.out.mkdir(parents=True, exist_ok=True)
reader, t0 = (load_reader(args.translator) if args.translator else None), time.time()
packs = load_packs(args.packs) if args.packs else None
taps = glm_read({i["id"]: i["passage"] for i in items}, args.out, args.out.name)
ids = [i["id"] for i in items]
for p in args.gain_power:
    translate_packs(taps, ids, packs[0], packs[1], p, args.out / f"memory_p{p}") if packs else translate(taps, ids, reader, p, args.out / f"memory_p{p}")
    (args.out / f"memory_p{p}" / "memory").rename(args.out / f"memory{p}")
    (args.out / f"memory_p{p}").rmdir()
main = args.gain_power[0]
jobs = [{"id": it["id"], "question": it["question"], "answer": it["answer"], "passage": it["passage"],
         "memory": f"memory{main}/{it['id']}.npz", "wrong_memory": f"memory{main}/{ids[(n + 1) % len(ids)]}.npz",
         "memories": {f"drift_p{p}": f"memory{p}/{it['id']}.npz" for p in args.gain_power[1:]}} for n, it in enumerate(items)]
answers = qwen_answer(jobs, args.out, args.out.name, args.max_new)
conditions = ["no_memory", "drift", "wrong_memory", "own_kv_only", "text_in_prompt"] + [f"drift_p{p}" for p in args.gain_power[1:]]
hit = lambda row, c: row["answer"].lower() in row[c]["text"].lower()
rows = answers["results"]
em = {c: sum(hit(r, c) for r in rows) / len(rows) for c in conditions}
lp = {c: sum(r[c]["answer_logprob"] for r in rows) / len(rows) for c in conditions}
wins, losses = sum(hit(r, "drift") and not hit(r, "wrong_memory") for r in rows), sum(hit(r, "wrong_memory") and not hit(r, "drift") for r in rows)
p_sign = sum(comb(wins + losses, k) for k in range(wins, wins + losses + 1)) / 2 ** (wins + losses) if wins + losses else 1.0
by_kind = {}
for it, r in zip(items, rows):
    k = by_kind.setdefault(it["kind"], {"n": 0, "drift": 0, "own_kv_only": 0}); k["n"] += 1; k["drift"] += hit(r, "drift"); k["own_kv_only"] += hit(r, "own_kv_only")
c1, c2 = (em["drift"] - em["wrong_memory"] >= 0.25 and p_sign < 0.05), em["drift"] >= 0.5 * em["own_kv_only"]
report = {"kind": "VALIDATION (hyperparameter selection, not evidence)" if args.validation else "PREREGISTERED EVALUATION",
          "preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest() if args.prereg else None,
          "translator_sha256": [x.sha256 for x in packs] if packs else reader.sha256, "translator_meta": packs[1].meta if packs else reader.meta, "gain_power": args.gain_power, "items": len(rows),
          "exact_match": em, "mean_answer_logprob": lp, "discordant": {"drift_only": wins, "wrong_only": losses, "sign_test_p": p_sign}, "by_kind": by_kind,
          "C1_content_specific": c1, "C2_useful": c2, "verdict": None if args.validation else ("PASSED" if c1 and c2 else "PARTIAL" if c1 else "FAILED"),
          "seconds": round(time.time() - t0, 1), "rows": [{**{k: it[k] for k in ("id", "kind", "question", "answer")}, **{c: r[c]["text"] for c in conditions}} for it, r in zip(items, rows)]}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
