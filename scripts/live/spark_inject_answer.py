"""Spark host: GLM answers questions with foreign memory written into its cache by DriftGlm53Connector.
Jobs: [{id, question, answer?, passage? (controls only), memories: {condition: name-of-npz-already-in tp-inject}}].
Stdlib only. The API key comes from the environment and is never printed."""
import argparse, json, os, time, urllib.request, uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--copies", type=float, default=1.0, help="how many copies of the memory fill the placeholder span (a log(copies) attention prior)")
parser.add_argument("--placeholder-id", type=int, required=True)
parser.add_argument("--max-new", type=int, default=80)
parser.add_argument("--skip-controls", action="store_true")
parser.add_argument("--system", default="", help="system prompt for memory conditions (keep the whole own prompt under 128 tokens)")
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
KEY, INJECT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff/tp-inject")


def post(path, body):
    req = urllib.request.Request(args.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def chat_ids(question, context=None, system=""):
    user = question if context is None else f"{context}\n\n{question}"
    return post("/tokenize", {"model": args.model, "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}], "add_generation_prompt": True,
                              "chat_template_kwargs": {"enable_thinking": False}})["tokens"]


def plan(own_tokens, rows):
    """Placeholder layout for this build's scheduler: the first prefill step of a prompt of L tokens ends at
    floor(L/64)*64 - 64. Memory fills [0, B); own prompt starts at S; the step must end inside [B, S]."""
    if own_tokens >= 128:
        raise ValueError("own prompt of 128+ tokens: the first engine step would run past the memory span on this build")
    memory = -(-int(round(args.copies * rows)) // 64) * 64
    return memory, memory if own_tokens >= 64 else memory + 64


def complete(ids, inject=None, rows=0, pad_only=False):
    prompt = ids
    if inject or pad_only:
        memory, own_start = plan(len(ids), rows)
        prompt = [args.placeholder_id] * own_start + ids
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_new, "temperature": 0, "cache_salt": f"drift:{uuid.uuid4().hex}",
            "vllm_xargs": {"skip_writing_prefix_cache": 1}}
    if inject:
        body["kv_transfer_params"] = {"drift_inject": inject, "drift_tokens": memory, "drift_own_start": own_start}
        for stale in INJECT.glob(f"{inject}.rank*.*"):
            if stale.suffix in (".done", ".error"):
                stale.unlink()
    started = time.time()
    out = {"text": post("/v1/completions", body)["choices"][0]["text"], "seconds": round(time.time() - started, 2)}
    if inject:
        out["layout"] = {"memory_positions": memory, "own_start": own_start, "own_tokens": len(ids)}
        done, error = INJECT / f"{inject}.rank0.done", INJECT / f"{inject}.rank0.error"
        out["inject"] = json.loads(done.read_text()) if done.exists() else {"error": json.loads(error.read_text())["error"] if error.exists() else "no report from the connector"}
    return out


results, started = [], time.time()
if args.out.exists():                                             # resume after a dropped link: keep what was already answered
    results = json.loads(args.out.read_text())["results"]
done = {r["id"] for r in results}
for job in json.loads(args.jobs.read_text()):
    if job["id"] in done:
        continue
    ids = chat_ids(job["question"], system=args.system)
    row = {"id": job["id"], "question": job["question"], "answer": job.get("answer")}
    if not args.skip_controls:
        row["no_memory"] = complete(ids)
    for condition, spec in (job.get("memories") or {}).items():
        if not args.skip_controls:
            row.setdefault("placeholders_only", complete(ids, rows=spec["rows"], pad_only=True))
        row[condition] = complete(ids, inject=spec["name"], rows=spec["rows"])
    if job.get("passage"):
        row["text_in_prompt"] = complete(chat_ids(job["question"], job["passage"]))
    results.append(row)
    args.out.write_text(json.dumps({"seconds": round(time.time() - started, 1), "results": results}, indent=2) + "\n")
print(json.dumps({"jobs": len(results), "seconds": round(time.time() - started, 1)}))
