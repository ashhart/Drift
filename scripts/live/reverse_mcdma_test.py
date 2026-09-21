"""Bounded live test: does GLM USE memory that Qwen published over MCDMA? (coordinator; EXPLORATORY, development passages)
Per item, three GLM sessions with the same prompt shape: memory (Qwen read the right passage), wrong_memory (another
passage), no_memory (nothing published; the span keeps its placeholders). Memory payloads travel Studio -> both Sparks only
through the dedicated MCDMA targets; ssh carries control. Every publication must be confirmed by BOTH ranks' connector
receipts, bound to the same session, sequence and SHA-256, separately from the mailbox acknowledgements."""
import argparse, json, subprocess, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import OMLX_PY, SPARK, SPARK_PEERS, STUDIO, sh
from questions_v3 import PRIORITY, QUESTION

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--limit", type=int, default=8)
parser.add_argument("--skip", type=int, default=0)
parser.add_argument("--copies", type=int, default=12)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; you were never shown it as text. @@DRIFT@@ "
        "Answer the user's question from what you recall from that shared memory. Give the value directly and keep the answer to one short sentence. "
        "If the shared memory does not contain what is needed, say plainly that it is missing rather than guessing.")
items = []
for line in args.passages.read_text().splitlines():
    r = json.loads(line); facts = {f["kind"]: f["value"] for f in r["facts"]}
    kind = next((k for k in PRIORITY if k in facts), None)
    if kind:
        items.append({"text": r["text"], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"]), "answer": facts[kind], "kind": kind})
items = items[args.skip: args.limit]
args.out.mkdir(parents=True, exist_ok=True); run = f"rev-{uuid.uuid4().hex[:8]}"
hosts = (SPARK, *SPARK_PEERS)
for host in hosts:
    sh("scp", "-q", "drift/serving/mcdma_mailbox.py", "scripts/mcdma_target/spark_bridge.py", "scripts/mcdma_target/npz_tile.py", "scripts/mcdma_target/spark_glm_session.py", "scripts/mcdma_target/glm_completion_sse.py", f"{host}:/root/drift-live/mcdma-target/")
sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift-frontier/"); sh("rsync", "-a", "local/live/stacked3_rev.npz", f"{STUDIO}:drift-frontier/local/live/")
bridges = [subprocess.Popen(["ssh", "-o", "ServerAliveInterval=15", host, f"python3 /root/drift-live/mcdma-target/spark_bridge.py --rank {rank} {'--head' if rank == 0 else ''} --seconds 3000"], stdout=subprocess.PIPE, text=True)
           for rank, host in enumerate(hosts)]
for b in bridges:
    print("bridge", b.stdout.readline().strip(), flush=True)
worker = subprocess.Popen(["ssh", "-o", "ServerAliveInterval=15", STUDIO, f"cd ~/drift-frontier && {OMLX_PY} scripts/live/studio_mcdma_reverse.py 2>out/mcdma_reverse_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
print("studio", worker.stdout.readline().strip(), flush=True)


def call(**cmd):
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok"):
        raise RuntimeError(reply)
    return reply


rows = []
try:
    for n, it in enumerate(items):
        other = items[(n + 1) % len(items)]
        right, wrong = call(op="prepare", key=f"right{n}", text=it["text"], copies=args.copies), call(op="prepare", key=f"wrong{n}", text=other["text"], copies=args.copies)
        span_rows = max(right["rows"], wrong["rows"])
        row = {"question": it["question"], "answer": it["answer"], "kind": it["kind"], "prepare": right}
        for condition, key in (("memory", f"right{n}"), ("wrong_memory", f"wrong{n}"), ("no_memory", None)):
            session = f"{run}-{n}-{condition.replace('_', '')}"
            messages = json.dumps([{"role": "system", "content": LINK}, {"role": "user", "content": it["question"]}])
            subprocess.run(["ssh", SPARK, f"cat > /root/drift-live/mcdma-target/{session}.messages.json"], input=messages, text=True, check=True)
            glm = subprocess.Popen(["ssh", SPARK, f"/root/drift-live/spark_run.sh mcdma-target/spark_glm_session.py --session {session} --rows {span_rows} --messages /root/drift-live/mcdma-target/{session}.messages.json"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            ready = json.loads(glm.stdout.readline())
            result = {"glm_prompt": ready}
            t0 = time.perf_counter()
            if key:
                result["delivered"] = call(op="deliver", key=key, session=session, sequence=0)
            t_go = time.perf_counter(); glm.stdin.write("go\n"); glm.stdin.flush()
            if key:
                result["applied"] = call(op="confirm", key=key); result["go_to_applied_both_ranks_s"] = round(time.perf_counter() - t_go, 4)
            answer = json.loads(glm.stdout.readline()); glm.wait(timeout=60)
            result.update(glm=answer, correct=it["answer"].lower() in answer["text"].lower(), deliver_wall_s=round(t_go - t0, 4))
            row[condition] = result
            for host in hosts:                                                  # the Sparks run with a few GB free: never let publication files pile up in their /dev/shm
                subprocess.run(["ssh", host, f"rm -rf /dev/shm/glm53-handoff/tp-live-in/{session} /dev/shm/glm53-handoff/tp-live-out/{session} /root/drift-live/mcdma-target/{session}.messages.json"], capture_output=True)
            print(n, condition, result["correct"], "|", answer["text"][:70].replace("\n", " "), "|", {k: result.get(k) for k in ("deliver_wall_s", "go_to_applied_both_ranks_s")}, flush=True)
        rows.append(row)
        (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY: reverse Drift over MCDMA, bounded live test", "run": run, "rows": rows}, indent=1))
finally:
    try:
        worker.stdin.write('{"op": "quit"}\n'); worker.stdin.flush()
    except Exception:
        pass
    for b in bridges:
        b.terminate()
    for host in hosts:
        subprocess.run(["ssh", host, "pkill -TERM -f '[s]park_bridge.py'; rm -rf /dev/shm/glm53-handoff/tp-live-in/" + run + "-* /dev/shm/glm53-handoff/tp-live-out/" + run + "-* /root/drift-live/mcdma-target/" + run + "-*.messages.json"], capture_output=True)
summary = {c: sum(r[c]["correct"] for r in rows) for c in ("memory", "wrong_memory", "no_memory")}
print(json.dumps({"items": len(rows), "correct": summary}))
