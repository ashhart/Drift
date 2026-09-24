"""Capture GLM's recurrent-layer inputs over training passages, the targets for a translated recurrent state.

Runs on the head Spark through spark_run.sh, against a server started with --enforce-eager and drift_hidden_capture.
Each passage is read in the framing the gate and the loop use: the chat template's tokens before @@DRIFT@@, then the
passage's own GLM tokens. One request per passage records, on rank 0, every KDA layer's input at the passage's rows.
The same request exports GLM's MLA latents. Each passage is saved as one npz: the passage's token character offsets,
h{layer} float16 [tokens, 4096] for the KDA layers and l{layer} float16 [tokens, 512] for the MLA layers.
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

LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; "
        "you were never shown it as text. @@DRIFT@@ Answer the user's question from what you recall from that shared memory. "
        "Give the answer directly.")
parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path, required=True)
parser.add_argument("--split", choices=["train", "val"], required=True)
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host, whose export copy is removed after each passage")
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
CAPTURE = ROOT / "drift-capture"
args.out.mkdir(parents=True, exist_ok=True)


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def head_tokens() -> list[int]:
    tokenized = post("/tokenize", {"model": args.model, "messages": [{"role": "system", "content": LINK}, {"role": "user", "content": "?"}],
                                   "add_generation_prompt": True, "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False}})
    text, spans = "", []
    for piece in tokenized["token_strs"]:
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    a = text.index("@@DRIFT@@")
    return tokenized["tokens"][:next(i for i, (x, y) in enumerate(spans) if y > a)]


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


def passage_tokens(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    tokenized = post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False, "return_token_strs": True})
    offsets, at = [], 0
    for piece in tokenized["token_strs"]:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    if at != len(text) or not text.isascii():
        return None, None                                            # byte-level pieces map one to one onto ASCII only; skip the rest
    return tokenized["tokens"], offsets


HEAD = head_tokens()
passages = {}
for triple in json.loads(args.triples.read_text())[args.split]:
    passages.setdefault(triple["pid"], triple["passage"])
done, started, written, skipped = {p.stem for p in args.out.glob("*.npz")}, time.time(), 0, 0
for pid, text in list(passages.items())[: args.limit]:
    if pid in done:
        continue
    ids, offsets = passage_tokens(text)
    if ids is None or len(ids) > 512:                                # unalignable, or more rows than one capture holds
        skipped += 1
        continue
    name, rows = f"hid-{uuid.uuid4().hex[:12]}", list(range(len(HEAD), len(HEAD) + len(ids)))
    post("/v1/completions", {"model": args.model, "prompt": HEAD + ids, "max_tokens": 1, "temperature": 0, "cache_salt": f"drift:{uuid.uuid4().hex}",
                             "kv_transfer_params": {"drift_capture": name, "drift_capture_rows": rows, "glm53_handoff": True, "handoff_id": name}})
    deadline = time.monotonic() + 60
    while not (CAPTURE / f"{name}.rank0.done").exists():
        if (CAPTURE / f"{name}.rank0.error").exists() or time.monotonic() > deadline:
            raise SystemExit(f"{pid}: capture failed: {(CAPTURE / f'{name}.rank0.error').read_text()[:300] if (CAPTURE / f'{name}.rank0.error').exists() else 'timeout'}")
        time.sleep(0.05)
    while not (ROOT / name / "rank0.ready").exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{pid}: no latent export")
        time.sleep(0.05)
    latents = read_latents(export_bytes(name))
    saved = np.load(CAPTURE / f"{name}.hidden.npz")
    if saved["positions"].tolist() != rows:
        raise SystemExit(f"{pid}: captured rows do not match the request")
    np.savez(args.out / f"{pid}.npz", offsets=np.asarray(offsets, np.int32), head=np.int32(len(HEAD)),
             **{k: saved[k] for k in saved.files if k.startswith("h")},
             **{f"l{layer}": value[rows[0]:rows[-1] + 1].astype(np.float16) for layer, value in latents.items()})
    subprocess.run(["rm", "-rf", str(ROOT / name)], check=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, f"rm -rf {ROOT / name} {CAPTURE / name}.rank1.done"], check=True, timeout=60)
    for suffix in ("hidden.npz", "rank0.done", "rank1.done"):
        (CAPTURE / f"{name}.{suffix}").unlink(missing_ok=True)
    written += 1
    if written % 25 == 0:
        print(json.dumps({"done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"split": args.split, "written": written, "skipped": skipped, "seconds": round(time.time() - started, 1)}))
