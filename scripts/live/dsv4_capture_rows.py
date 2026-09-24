"""Capture DeepSeek V4's compressed cache rows over training passages, for translators to and from Qwen and GLM.

Runs on the head Spark next to raw_rows.py, against a server with the raw-row connector. Each passage is read in the
layout the Drift gate uses, aligned to whole 256-token cache pages: an opening padded to 256 tokens, the passage padded to
a multiple of 256 with newlines, then four newlines. A page stores its entries first and their scales at its end, so
only whole pages can be decoded. A tap request reads the span's compressed MLA and indexer entries back; rank 0's copy is kept, since MLA
keeps one latent head on every rank. Each passage is saved as one npz: the passage tokens' character offsets, the span,
and the tapped rows in raw_rows' format. Only ASCII passages are kept, where byte-level tokens map one to one onto
characters.
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
from raw_rows import load

parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path, required=True)
parser.add_argument("--split", choices=["train", "val"], required=True)
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host, whose tap copy is removed")
parser.add_argument("--api", default="http://127.0.0.1:8888")
parser.add_argument("--path", type=Path, default=Path("/dev/shm/drift-rows"))
args = parser.parse_args()
KEY, ALIGN = os.environ.get("DRIFT_API_KEY", ""), 256            # whole cache pages: each page keeps its entries' scales at its end
args.out.mkdir(parents=True, exist_ok=True)


def post(path: str, body: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})}
    request = urllib.request.Request(args.api + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


MODEL = post("/v1/models")["data"][0]["id"]


def tokens(text: str, strings: bool = False) -> dict:
    return post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False, "return_token_strs": strings})


FILL = tokens("\n")["tokens"]
if len(FILL) != 1:
    raise SystemExit("the newline filler must be one token")
opening = tokens("Read this document, then answer the question after it.\n\nDocument:\n")["tokens"]
opening += FILL * ((-len(opening)) % ALIGN)
S = len(opening)
passages = {}
for triple in json.loads(args.triples.read_text())[args.split]:
    passages.setdefault(triple["pid"], triple["passage"])
done, started, written, skipped = {p.stem for p in args.out.glob("*.npz")}, time.time(), 0, 0
for pid, text in list(passages.items())[: args.limit]:
    if pid in done:
        continue
    tokenized = tokens(text, strings=True)
    ids, offsets, at = tokenized["tokens"], [], 0
    for piece in tokenized["token_strs"]:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    if at != len(text) or not text.isascii():
        skipped += 1
        continue
    T = len(ids) + (-len(ids)) % ALIGN
    name = f"cap-{uuid.uuid4().hex[:10]}"
    body = {"model": MODEL, "prompt": opening + ids + FILL * (T - len(ids) + 4), "max_tokens": 1, "temperature": 0,
            "cache_salt": uuid.uuid4().hex, "kv_transfer_params": {"drift_tap": name, "drift_tokens": T, "drift_span_start": S}}
    post("/v1/completions", body)
    tap = args.path / "drift-tap" / f"{name}.rank0.npz"
    deadline = time.monotonic() + 60
    while not tap.exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{pid}: tap not written")
        time.sleep(0.1)
    rows = load(tap)
    manifest = [{"name": n, "ratio": r} for n, (r, _) in sorted(rows.items())]
    np.savez(args.out / f"{pid}.npz", offsets=np.asarray(offsets, np.int32), span=np.asarray([S, T, len(ids)], np.int32),
             manifest=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8),
             **{f"r{i}": rows[entry["name"]][1] for i, entry in enumerate(manifest)})
    tap.unlink()
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, f"rm -f {args.path / 'drift-tap' / name}.rank1.npz"], check=True, timeout=60)
    written += 1
    if written % 50 == 0:
        print(json.dumps({"done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"split": args.split, "written": written, "skipped": skipped, "seconds": round(time.time() - started, 1)}))
