"""Test A REPLICATED over MCDMA (coordinator). Same passages and questions as scripts/live/qa_v4.py and the same frozen
artefacts; memory moves Spark -> Studio by RDMA through the owner's handoffd instead of files over ssh. Only control messages
and the jobs list (text and questions, which Test A's text control needs anyway) use ssh. Resumable per chunk."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizers import Tokenizer
from livelib import OMLX_PY, SPARK, STUDIO, passage_id, sh
from questions_v3 import NUMERIC, PRIORITY, QUESTION
from drift.eval.paired_stats import clustered_bootstrap, passage_differences, sign_flip_p

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--frozen", type=Path, required=True)
parser.add_argument("--original", type=Path, help="Test A's report.json, for the row-by-row comparison")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int, default=120)
parser.add_argument("--chunk", type=int, default=24)
args = parser.parse_args()
frozen = json.loads(args.frozen.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key, path in frozen["artifacts"].items():
    if sha(path) != frozen["sha256"][key]:
        raise SystemExit(f"frozen artifact {key} does not match its recorded sha256")
tok = Tokenizer.from_file("local/tok/glm/tokenizer.json")
passages = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line)
    if "questions" in r:                                                      # explicit questions: diagnostics outside Test A
        passages.append({"id": passage_id(r["text"]), "text": r["text"], "questions": r["questions"]})
        continue
    facts = {f["kind"]: f["value"] for f in r["facts"]}
    start = len(passages) % len(PRIORITY)
    kinds = [k for k in PRIORITY[start:] + PRIORITY[:start] if k in facts][:2]
    if len(kinds) == 2:
        passages.append({"id": passage_id(r["text"]), "text": r["text"], "questions": [{"kind": k, "answer": facts[k], "question": QUESTION[k].format(genre=r["genre"], thing=r["thing"])} for k in kinds]})
passages = passages[: args.limit]
args.out.mkdir(parents=True, exist_ok=True); run, t0 = args.out.name, time.time()
STUDIO_DIR = "drift-frontier"
sh("ssh", STUDIO, f"mkdir -p {STUDIO_DIR}/out/{run}"); sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:{STUDIO_DIR}/")
sh("ssh", SPARK, f"mkdir -p /root/drift-live/runs/{run}"); sh("scp", "-q", "scripts/live/spark_export_glm.py", "scripts/live/spark_run.sh", f"{SPARK}:/root/drift-live/")
rows_path = f"{STUDIO_DIR}/out/{run}/rows.jsonl"
done = set()
existing = subprocess.run(["ssh", STUDIO, f"cat {rows_path} 2>/dev/null"], capture_output=True, text=True).stdout
done = {json.loads(l)["passage"] for l in existing.splitlines() if l.strip()}
worker = subprocess.Popen(["ssh", "-o", "ServerAliveInterval=15", STUDIO, f"cd ~/{STUDIO_DIR} && {OMLX_PY} scripts/live/studio_mcdma_qa.py --out out/{run}/rows.jsonl 2>out/{run}/worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
while True:
    line = worker.stdout.readline()
    if not line:
        raise SystemExit("Studio worker exited before READY; see its worker.log")
    if line.startswith("READY"):
        break
exports = []
todo = [p for p in passages if p["id"] not in done]
for a in range(0, len(todo), args.chunk):
    chunk = todo[a:a + args.chunk]
    ids = {"records": [{"id": p["id"], "ids_glm": tok.encode(p["text"], add_special_tokens=False).ids} for p in chunk]}
    subprocess.run(["ssh", SPARK, f"cat > /root/drift-live/runs/{run}/ids.json"], input=json.dumps(ids), text=True, check=True)
    out = sh("ssh", SPARK, f"/root/drift-live/spark_run.sh spark_export_glm.py --ids /root/drift-live/runs/{run}/ids.json")
    remote = {json.loads(l)["id"]: json.loads(l) for l in out.splitlines() if l.startswith("{")}
    exports += list(remote.values())
    jobs = [{"id": p["id"], "peer": SPARK, "remote": remote[p["id"]]["remote"], "text": p["text"], "questions": p["questions"]} for p in chunk]
    subprocess.run(["ssh", STUDIO, f"cat > {STUDIO_DIR}/out/{run}/jobs_{a}.json"], input=json.dumps(jobs), text=True, check=True)
    worker.stdin.write(f"out/{run}/jobs_{a}.json\n"); worker.stdin.flush()
    reply = worker.stdout.readline()
    if not reply.startswith("DONE"):
        raise SystemExit(f"Studio worker failed on chunk {a}: {reply!r}")
    sh("ssh", SPARK, "rm -rf /dev/shm/glm53-handoff/drift-*")               # only this run's own export folders
    print("chunk", a, len(chunk), "passages", round(time.time() - t0), "s", flush=True)
worker.stdin.close(); worker.wait(timeout=120)
rows = [json.loads(l) for l in sh("ssh", STUDIO, f"cat {rows_path}").splitlines() if l.strip()]
conditions = ["no_memory", "drift_v3", "drift_v4", "wrong_memory", "own_kv", "text_in_prompt"]
hit = lambda r, c: r[c] is not None and r["answer"].lower() in r[c].lower()
em = lambda rs: {c: round(sum(hit(r, c) for r in rs if r[c] is not None) / max(1, sum(r[c] is not None for r in rs)), 4) for c in conditions}
outcomes = {}
for r in rows:
    outcomes.setdefault(r["passage"], []).append((hit(r, "drift_v4"), hit(r, "drift_v3")))
report = {"kind": "REPLICATION of Test A over MCDMA (same passages, questions and frozen artefacts; transport is the only intended change; translation runs on the Studio GPU)",
          "frozen_sha256": sha(args.frozen), "passages": len(outcomes), "questions": len(rows), "exact_match": em(rows),
          "exact_match_numeric": em([r for r in rows if r["kind"] in NUMERIC]), "v4_minus_v3": {"clustered_sign_flip_p": sign_flip_p(list(passage_differences(outcomes).values())), "bootstrap": clustered_bootstrap(outcomes)},
          "transport": {"bytes_total": sum(r["transport"]["bytes"] for r in rows[::2]), "median_rdma_loop_s": sorted(r["transport"]["rdma_loop_s"] for r in rows)[len(rows) // 2],
                        "median_gbit_s": sorted(r["transport"]["rdma_gbit_s"] for r in rows)[len(rows) // 2], "median_job_s": sorted(r["transport"]["job_s"] for r in rows)[len(rows) // 2]},
          "glm_export": {"median_request_s": sorted(e["request_s"] for e in exports)[len(exports) // 2] if exports else None}, "seconds": round(time.time() - t0, 1)}
if args.original and args.original.exists():
    old = {(r["passage"], r["kind"]): r for r in json.loads(args.original.read_text())["rows"]}
    both = [(r, old[(r["passage"], r["kind"])]) for r in rows if (r["passage"], r["kind"]) in old]
    report["versus_original_ssh_run"] = {"rows_compared": len(both), "original_exact_match": json.loads(args.original.read_text())["exact_match"],
                                         "same_verdict_rate": {c: round(sum(hit(n, c) == (o["answer"].lower() in o[c].lower()) for n, o in both) / max(1, len(both)), 4) for c in ("drift_v3", "drift_v4", "own_kv", "text_in_prompt")}}
report["rows"] = rows
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
