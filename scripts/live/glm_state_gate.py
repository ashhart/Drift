"""Own-cache gate for GLM: does its own attention rows plus its own recurrent state reproduce reading the text?

Runs on the head Spark through spark_run.sh, against a server started with the Drift scheduler. Per question, one
prompt frames a span with @@DRIFT@@ and four arms answer greedily:
  text        the passage's GLM tokens stand in the span.
  none        the span is empty.
  rows        the span is a reserve of exactly the passage's length; GLM's own exported latents of the passage
              arrive as the first publication, and the prefill stops at the reserve's end.
  rows_state  as rows, and each rank also writes its own exported recurrent state at that stop.
The export is GLM reading the head and the passage, so the state after it is what the text arm holds at that point.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import shlex
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
from glm53_handoff import read_latents

LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; "
        "you were never shown it as text. @@DRIFT@@ Answer the user's question from what you recall from that shared memory. "
        "Give the answer directly.")
parser = argparse.ArgumentParser()
parser.add_argument("--items", type=Path, required=True, help="json list of {id, passage, question, answer}")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host")
parser.add_argument("--max-new", type=int, default=80)
parser.add_argument("--memory", type=Path, help="translated memory per passage from translate_passages.py: adds t_rows and t_rows_state where a passage has one")
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
PUBLICATION_BYTES = 128 * 1024 * 1024                                # live_publication.MAX_BYTES on the Sparks
RESERVE_ROWS = 4096                                                  # glm_prefill_boundary's largest reserve


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def worker(command: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, command], input=data, capture_output=True,
                          check=True, timeout=120).stdout


def framing(question: str) -> tuple[list[int], list[int]]:
    tokenized = post("/tokenize", {"model": args.model, "messages": [{"role": "system", "content": LINK}, {"role": "user", "content": question}],
                                   "add_generation_prompt": True, "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False}})
    ids, text, spans = tokenized["tokens"], "", []
    for piece in tokenized["token_strs"]:
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    a = text.index("@@DRIFT@@")
    hit = [i for i, (x, y) in enumerate(spans) if x < a + len("@@DRIFT@@") and y > a]
    return ids[:hit[0]], ids[hit[-1] + 1:]


def complete(prompt: list[int], params: dict | None = None, max_new: int | None = None) -> dict:
    body = {"model": args.model, "prompt": prompt, "max_tokens": max_new or args.max_new, "temperature": 0,
            "cache_salt": f"drift:{uuid.uuid4().hex}", "vllm_xargs": {"skip_writing_prefix_cache": 1}}
    if params:
        body["kv_transfer_params"] = params
    started = time.time()
    choice = post("/v1/completions", body)["choices"][0]
    return {"text": choice["text"], "finish": choice.get("finish_reason"), "seconds": round(time.time() - started, 2)}


def export_bytes(name: str) -> bytes:
    """Rank 0's export, from its file or from the daemon's arena as its ready file says."""
    ready = json.loads((ROOT / name / "rank0.ready").read_text())
    if ready.get("mode") != "arena":
        return (ROOT / name / "rank0.bin").read_bytes()
    arena = ROOT / "arena-local0.bin"
    if os.stat(arena).st_ino != ready["arena_inode"]:
        raise RuntimeError("rank 0's export arena was replaced")
    with arena.open("rb") as handle:
        handle.seek(int(ready["offset"]))
        return handle.read(int(ready["length"]))


def fresh(name: str, limit: float = 90) -> None:
    """Refuse before posting: a failed state write takes the engine down, so both exports must be present and young."""
    probe = (f"import json, os, sys, time; r = json.load(open(sys.argv[1] + '/rank' + sys.argv[2] + '.ready')); "
             f"ok = r.get('mode') == 'arena' or os.path.exists(sys.argv[1] + '/rank' + sys.argv[2] + '.bin'); "
             f"print('ok' if ok and time.time() - float(r.get('created', 0)) < {limit} else 'stale')")
    local = subprocess.run(["python3", "-c", probe, str(ROOT / name), "0"], capture_output=True, text=True, check=True).stdout.strip()
    remote = worker(f"python3 -c {shlex.quote(probe)} {shlex.quote(str(ROOT / name))} 1").decode().strip()
    if (local, remote) != ("ok", "ok"):
        raise RuntimeError(f"export {name} is not usable on both ranks: {local}, {remote}")


def wait(local: list[Path], remote: list[Path], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        there = not remote or worker(" && ".join(f"test -e {shlex.quote(str(p))}" for p in remote) + " && echo yes || true").strip() == b"yes"
        if there and all(p.exists() for p in local):
            return
        time.sleep(0.2)
    raise TimeoutError(f"not on both ranks: {[p.name for p in local + remote]}")


def stage(session: str, name: str, arrays: dict) -> None:
    """One npz in the session's inbox on both hosts, renamed into place only when complete."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    folder = ROOT / "tp-live-in" / session
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f".{name}.tmp").write_bytes(buffer.getvalue())
    os.replace(folder / f".{name}.tmp", folder / name)
    remote = shlex.quote(str(folder))
    worker(f"mkdir -p {remote} && cat > {remote}/.{name}.tmp && mv {remote}/.{name}.tmp {remote}/{name}", buffer.getvalue())


def publish(session: str, rows: dict[int, np.ndarray]) -> None:
    stage(session, "000000.npz", {f"l{layer}": value.astype(np.float32) for layer, value in rows.items()})


def linked(session: str, head: list[int], tail: list[int], rows: dict[int, np.ndarray], blob: str | None, inputs: dict | None = None) -> dict:
    n = next(iter(rows.values())).shape[0]
    if n > RESERVE_ROWS:                                                  # the scheduler's boundary check stops the engine past this
        return {"skipped": f"reserve of {n} rows; the receiver takes up to {RESERVE_ROWS}"}
    rows_bytes = sum(v.size * 4 for v in rows.values())                  # published as float32
    if rows_bytes > PUBLICATION_BYTES:                                    # the receiver refuses such a publication and poisons the engine
        return {"skipped": f"rows publication of {rows_bytes} bytes; the receiver takes up to {PUBLICATION_BYTES}"}
    publish(session, rows)
    params = {"drift_session": session, "drift_reserve": n, "drift_reserve_start": len(head), "drift_prefill_boundary": len(head) + n, "drift_tap": False}
    if inputs is not None:
        stage(session, "state.npz", {f"h{layer}": value for layer, value in inputs.items()})
        params["drift_state_rows"] = True                             # advance the head's state by the memory's layer inputs
    if blob:
        fresh(blob)
        params["drift_state_blob"] = blob
    out = complete(head + [args.placeholder_id] * n + tail, params)
    receipts = [ROOT / "tp-live-out" / session / f"ack.000000.rank{r}.json" for r in (0, 1)]
    marks = [ROOT / "tp-live-out" / session / f"state.rank{r}.json" for r in (0, 1)] if blob else []
    wait([receipts[0]] + marks[:1], [receipts[1]] + marks[1:], timeout=30)
    if blob:
        out["state"] = [json.loads(marks[0].read_text()), json.loads(worker(f"cat {shlex.quote(str(marks[1]))}"))]
    return out


def clean(names: list[str]) -> None:
    paths = " ".join(shlex.quote(str(p)) for n in names for p in (ROOT / n, ROOT / "tp-live-in" / n, ROOT / "tp-live-out" / n))
    subprocess.run(f"rm -rf {paths}", shell=True, check=True)
    worker(f"rm -rf {paths}")


results = json.loads(args.out.read_text())["results"] if args.out.exists() else []
done, started = {r["id"] for r in results}, time.time()
for item in json.loads(args.items.read_text()):
    if item["id"] in done:
        continue
    head, tail = framing(item["question"])
    passage = post("/tokenize", {"model": args.model, "prompt": item["passage"], "add_special_tokens": False})["tokens"]
    row = {"id": item["id"], "question": item["question"], "answer": item.get("answer"), "passage_tokens": len(passage), "head": len(head)}
    row["text"] = complete(head + passage + tail)
    row["none"] = complete(head + tail)
    blob, tag = f"own-{uuid.uuid4().hex[:10]}", uuid.uuid4().hex[:8]
    complete(head + passage, {"glm53_handoff": True, "handoff_id": blob}, max_new=1)
    wait([ROOT / blob / "rank0.ready"], [ROOT / blob / "rank1.ready"])
    latents = read_latents(export_bytes(blob))
    rows = {layer: value[len(head): len(head) + len(passage)] for layer, value in latents.items()}
    sessions = [f"gate-{tag}-rows", f"gate-{tag}-state"]
    try:
        row["rows"] = linked(sessions[0], head, tail, rows, None)
        row["rows_state"] = linked(sessions[1], head, tail, rows, blob)
    finally:
        clean([blob] + sessions)
    memory_path = args.memory / f"{item['id'].rsplit('-', 1)[0]}.npz" if args.memory else None
    if memory_path is not None and memory_path.exists():                  # a sender may lack a passage, such as DeepSeek's non-ASCII ones
        memory = np.load(memory_path)
        translated = {int(k[1:]): memory[k].astype(np.float32) for k in memory.files if k[0] == "l" and k[1:].isdigit()}
        inputs = {int(k[1:]): memory[k] for k in memory.files if k[0] == "h" and k[1:].isdigit()}
        base, sessions = f"head-{uuid.uuid4().hex[:10]}", [f"gate-{tag}-trows", f"gate-{tag}-tstate"]
        try:
            row["t_rows"] = linked(sessions[0], head, tail, translated, None)
            state_bytes = sum(v.nbytes for v in inputs.values())
            if not inputs or state_bytes > PUBLICATION_BYTES:                # the receiver refuses such a publication and poisons the engine
                row["t_rows_state"] = {"skipped": f"state publication of {state_bytes} bytes; the receiver takes 1 to {PUBLICATION_BYTES}"}
            else:
                complete(head, {"glm53_handoff": True, "handoff_id": base}, max_new=1)   # the state after the head alone
                wait([ROOT / base / "rank0.ready"], [ROOT / base / "rank1.ready"])
                row["t_rows_state"] = linked(sessions[1], head, tail, translated, base, inputs)
        finally:
            clean([base] + sessions)
    results.append(row)
    args.out.write_text(json.dumps({"seconds": round(time.time() - started, 1), "results": results}, indent=2) + "\n")
    print(json.dumps({"id": item["id"], **{arm: row[arm].get("text", row[arm].get("skipped", ""))[:60] for arm in ("text", "rows", "rows_state", "t_rows", "t_rows_state") if arm in row}}), flush=True)
print(json.dumps({"items": len(results), "seconds": round(time.time() - started, 1)}))
