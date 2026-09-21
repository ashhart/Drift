"""Fast A/B of reader-side variants on the Studio drift worker (model stays loaded). VALIDATION tool, not evidence.
  --memory name=dir   a folder of <passage id>.npz Qwen entries (several allowed)
Each memory is asked with and without the link system prompt. Exact-match scoring as in qa_eval.py."""
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import OMLX_PY, STUDIO, passage_id, sh

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--memory", action="append", default=[])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--max-new", type=int, default=160)
parser.add_argument("--prompts", default="plain,link")
parser.add_argument("--kinds", default="", help="restrict questions to these fact kinds (comma-separated), one item per matching fact")
args = parser.parse_args()
LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; you were never shown it as text. "
        "Answer from what you recall of that document. If the shared memory does not contain what is needed, say that it is missing.")
src = (Path(__file__).resolve().parent / "qa_eval.py").read_text()
exec(src[src.index("PRIORITY = ["):src.index("glm, qwen = Tokenizer")])
items = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line); facts = {f["kind"]: f["value"] for f in r["facts"]}
    wanted = [k for k in args.kinds.split(",") if k]
    for kind in ([k for k in wanted if k in facts] if wanted else [next((k for k in PRIORITY[len(items) % len(PRIORITY):] + PRIORITY if k in facts), None)]):
      if kind:
        items.append({"id": passage_id(r["text"]), "kind": kind, "answer": facts[kind], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"])})
args.out.mkdir(parents=True, exist_ok=True)
run = args.out.name
remote = f"local/studio/runs/{run}"
sh("ssh", STUDIO, f"mkdir -p drift/{remote}")
memories = dict(m.split("=", 1) for m in args.memory)
for name, folder in memories.items():
    sh("rsync", "-a", folder.rstrip("/") + "/", f"{STUDIO}:drift/{remote}/{name}/")
sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
worker = subprocess.Popen(["ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_drift_worker.py 2>local/studio/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def call(**cmd):
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok", reply.get("ready")):
        raise RuntimeError(reply)
    return reply


json.loads(worker.stdout.readline())
partial = args.out / "rows.partial.jsonl"                           # resumable: a dropped ssh link costs one item, not the run
rows = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
score = {}
for r in rows:
    for key in [k for k in r if "/" in k]:
        score[key] = score.get(key, 0) + (r["answer"].lower() in r[key].lower())
done = {(r["id"], r["kind"]) for r in rows}
for it in items:
    if (it["id"], it["kind"]) in done:
        continue
    row = dict(it)
    for name in memories:
        for prompt in args.prompts.split(","):
            call(op="start"); call(op="append", memory=f"{remote}/{name}/{it['id']}.npz"); call(op="extend", chat=it["question"], system=LINK if prompt == "link" else None)
            text = call(op="generate", tokens=args.max_new)["text"]
            key = f"{name}/{prompt}"; row[key] = text; score[key] = score.get(key, 0) + (it["answer"].lower() in text.lower())
    rows.append(row)
    with partial.open("a") as sink:
        sink.write(json.dumps(row) + "\n")
worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush()
report = {"items": len(items), "exact_match": {k: round(v / len(items), 3) for k, v in score.items()}, "rows": rows}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["exact_match"], indent=1))
for k in score:
    print(k, "misses:", [(r["kind"], r["answer"]) for r in rows if r["answer"].lower() not in r[k].lower()])
