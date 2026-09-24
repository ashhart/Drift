"""Export GLM's MLA latents for passages placed as its own assistant turn, the context its live taps come from.

Runs on the head Spark through spark_run.sh. Forward taps carry latents of text GLM wrote, not text it read, so the
GLM-to-Qwen translators are fitted on the same kind: a chat prefix with a system prompt and a user request, then the
passage as GLM's own reply. One export per passage; the assistant span's latents are saved with the passage tokens'
character offsets. Training passages must be ASCII, where byte-level tokens map one to one onto characters; test
items are kept either way and marked.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
from glm53_handoff import read_latents

SYSTEM = "You are a helpful assistant working with a partner model on the user's request."
REQUEST = "Write the requested document."
parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path, help="training triples: passages by pid")
parser.add_argument("--split", choices=["train", "val"])
parser.add_argument("--items", type=Path, help="alternatively, a json list of {id, passage}")
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host, whose export copy is removed")
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
if bool(args.items) == bool(args.triples and args.split):
    parser.error("give --items, or --triples with --split")
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
args.out.mkdir(parents=True, exist_ok=True)


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def export_bytes(name: str) -> bytes:
    """Rank 0's export, from its file or from the daemon's arena as its ready file says."""
    ready = json.loads((ROOT / name / "rank0.ready").read_text())
    if ready.get("mode") != "arena":
        return (ROOT / name / "rank0.bin").read_bytes()
    arena = ROOT / "arena-local0.bin"
    if os.stat(arena).st_ino != ready["arena_inode"]:
        raise SystemExit("rank 0's export arena was replaced")
    with arena.open("rb") as handle:
        handle.seek(int(ready["offset"]))
        return handle.read(int(ready["length"]))


PREFIX = post("/tokenize", {"model": args.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": REQUEST}],
                            "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
if args.items:
    passages = {item["id"].split("-")[0]: item["passage"] for item in json.loads(args.items.read_text())}
else:
    passages = {}
    for triple in json.loads(args.triples.read_text())[args.split]:
        passages.setdefault(triple["pid"], triple["passage"])
done, started, written, skipped = {p.stem for p in args.out.glob("*.npz")}, time.time(), 0, 0
for pid, text in list(passages.items())[: args.limit]:
    if pid in done:
        continue
    tokenized = post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False, "return_token_strs": True})
    ids, offsets, at = tokenized["tokens"], [], 0
    for piece in tokenized["token_strs"]:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    aligned = at == len(text) and text.isascii()
    if not aligned and not args.items:                                # training rows need exact offsets; test items only translate
        skipped += 1
        continue
    name = f"gen-{uuid.uuid4().hex[:10]}"
    post("/v1/completions", {"model": args.model, "prompt": PREFIX + ids, "max_tokens": 1, "temperature": 0, "cache_salt": f"drift:{uuid.uuid4().hex}",
                             "kv_transfer_params": {"glm53_handoff": True, "handoff_id": name}})
    deadline = time.monotonic() + 60
    while not (ROOT / name / "rank0.ready").exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{pid}: no export")
        time.sleep(0.05)
    latents = read_latents(export_bytes(name))
    np.savez(args.out / f"{pid}.npz", offsets=np.asarray(offsets, np.int32), aligned=np.bool_(aligned),
             **{f"l{layer}": value[len(PREFIX):len(PREFIX) + len(ids)].astype(np.float16) for layer, value in latents.items()})
    subprocess.run(["rm", "-rf", str(ROOT / name)], check=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, f"rm -rf {ROOT / name}"], check=True, timeout=60)
    written += 1
    if written % 50 == 0:
        print(json.dumps({"done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"written": written, "skipped": skipped, "seconds": round(time.time() - started, 1)}))
