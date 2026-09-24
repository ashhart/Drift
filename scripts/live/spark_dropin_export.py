"""The resident side of a drop-in: GLM reads each project context, and its cache export is left for an RDMA pull.

Runs on the head Spark through spark_run.sh. GLM reads a context in its reading frame, the system prompt up to @@DRIFT@@
then the project files, and the handoff connector exports the whole cache. The export stays in place for the Studio's
handoffd to pull over MCDMA; an export the daemon put in its arena is first copied into its own file, since handoffd
pulls files. Prints one JSON line per context: where the export is, its size, and the span the context's tokens take.
The connector computes a full read in one engine step: 60k tokens took 38 s here, and 121k never finished; --split-at
reads such a prompt as a full export of its first part and a delta export of the rest from the prefix cache.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; "
        "you were never shown it as text. @@DRIFT@@ Answer the user's question from what you recall from that shared memory. "
        "Give the answer directly.")                                      # the reading frame the forward translators were fitted on
parser = argparse.ArgumentParser()
parser.add_argument("--contexts", type=Path, required=True, help="json list of {id, text}")
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
parser.add_argument("--count", action="store_true", help="print each context's token count and export nothing")
parser.add_argument("--timeout", type=float, default=600, help="seconds allowed for one read, long contexts need more")
parser.add_argument("--split-at", type=int, default=0, help="read a longer prompt in two: its first N tokens with a full export, then the "
                    "whole prompt from the prefix cache with a delta export; N a multiple of the cache block (3,584 here)")
args = parser.parse_args()
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read())


tokenized = post("/tokenize", {"model": args.model, "messages": [{"role": "system", "content": LINK}, {"role": "user", "content": "?"}],
                               "add_generation_prompt": True, "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False}})
text, spans = "", []
for piece in tokenized["token_strs"]:
    spans.append((len(text), len(text) + len(piece)))
    text += piece
marker = text.index("@@DRIFT@@")
head = tokenized["tokens"][:next(i for i, (x, y) in enumerate(spans) if y > marker)]


def read(prompt: list[int], salt: str, export_from: int) -> Path:
    """One handoff read; the export's path. A delta read (export_from > 0) resumes from the prefix cache."""
    name = f"dropin-{uuid.uuid4().hex[:10]}"
    params = {"glm53_handoff": True, "handoff_id": name, **({"export_from": export_from} if export_from else {})}
    post("/v1/completions", {"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "cache_salt": salt, "kv_transfer_params": params})
    ready_path, read_at = ROOT / name / "rank0.ready", time.time()
    while not ready_path.exists():
        if time.time() - read_at > 120:
            raise SystemExit(f"{name}: no export")
        time.sleep(0.01)
    ready, blob = json.loads(ready_path.read_text()), ROOT / name / "rank0.bin"
    if ready.get("mode") == "arena":                                     # copy the leased range out, then check the lease still held
        arena = ROOT / "arena-local0.bin"
        with arena.open("rb") as source, blob.open("wb") as sink:
            source.seek(int(ready["offset"]))
            sink.write(source.read(int(ready["length"])))
        if os.stat(arena).st_ino != ready["arena_inode"]:
            raise SystemExit(f"{name}: the arena was replaced during the copy")
    return blob


def header(blob: Path) -> dict:
    with blob.open("rb") as handle:
        handle.read(8)
        return json.loads(handle.read(int.from_bytes(handle.read(8), "little")))


for context in json.loads(args.contexts.read_text()):
    ids = post("/tokenize", {"model": args.model, "prompt": context["text"], "add_special_tokens": False})["tokens"]
    if args.count:
        print(json.dumps({"id": context["id"], "start": len(head), "tokens": len(ids)}), flush=True)
        continue
    prompt, salt, started = head + ids, f"drift:{uuid.uuid4().hex}", time.time()
    if args.split_at and len(prompt) > args.split_at:                   # one step per read: a long prompt in two halves
        blob = read(prompt[:args.split_at], salt, 0)
        first_s = time.time() - started
        delta = read(prompt, salt, args.split_at)
        hit = int(header(delta).get("hit", -1))
        if hit != args.split_at:
            raise SystemExit(f"{context['id']}: the second read hit {hit} cached tokens, not {args.split_at}")
        extra = {"remote_delta": str(delta), "delta_from": args.split_at, "delta_bytes": delta.stat().st_size, "first_read_s": round(first_s, 3)}
    else:
        blob, extra = read(prompt, salt, 0), {}
    print(json.dumps({"id": context["id"], "remote": str(blob), "bytes": blob.stat().st_size, "start": len(head), "tokens": len(ids),
                      "glm_read_s": round(time.time() - started, 3), **extra}), flush=True)
