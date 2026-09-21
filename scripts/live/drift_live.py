"""Live exchange: GLM-5.3 (Sparks, vLLM) reads passages; its cache latents are exported by the running
server, translated, and written into Qwen3.8's cache (Studio, oMLX runtime), which then answers questions it was
never shown the text for.

  PYTHONPATH=. .venv/bin/python scripts/live/drift_live.py --cases scripts/live/cases.json --out local/live/demo \
      --translator local/live/stacked2.npz --gain-power 1.5 [--controls]
--controls also sends the passage text to the reader for the own-KV and text-in-prompt baselines."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import glm_read, load_reader, passage_id, qwen_answer, translate

parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--translator", type=Path, default=Path("local/live/stacked2.npz"))
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--controls", action="store_true")
parser.add_argument("--max-new", type=int, default=80)
args = parser.parse_args()
cases = json.loads(args.cases.read_text())
for c in cases:
    c["id"] = passage_id(c["passage"])
args.out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); taps = glm_read({c["id"]: c["passage"] for c in cases}, args.out, args.out.name)
t1 = time.time(); translate(taps, [c["id"] for c in cases], load_reader(args.translator), args.gain_power, args.out)
t2 = time.time()
jobs = [{"id": c["id"], "question": c["question"], "answer": c.get("answer"), "memory": f"memory/{c['id']}.npz",
         "wrong_memory": f"memory/{cases[(i + 1) % len(cases)]['id']}.npz" if len(cases) > 1 else None,
         **({"passage": c["passage"]} if args.controls else {})} for i, c in enumerate(cases)]
report = qwen_answer(jobs, args.out, args.out.name, args.max_new)
report["timing_seconds"] = {"glm_read_and_export": round(t1 - t0, 1), "translate": round(t2 - t1, 1), "qwen_load_and_answer": round(time.time() - t2, 1)}
(args.out / "answers.json").write_text(json.dumps(report, indent=2) + "\n")
for r in report["results"]:
    print(f"\nQ: {r['question']}   (reference: {r['answer']})")
    for name in ("no_memory", "drift", "wrong_memory", "own_kv_only", "text_in_prompt"):
        if name in r:
            lp = r[name].get("answer_logprob")
            print(f"  {name:>15}: {'' if lp is None else f'[{lp:7.2f}] '}{r[name]['text'][:200]!r}")
print("\ntiming:", report["timing_seconds"])
