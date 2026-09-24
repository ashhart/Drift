"""Export GLM's latents in the live session's prompt layout, where its forward taps come from.

Runs on the head Spark through spark_run.sh. The live prompt is GLM's chat template with a memory reserve where
@@DRIFT@@ stands: the head before it, the reserve holding the memory then placeholder tokens, and GLM's own tail after
it (the rest of the system prompt, the user's instructions, the assistant opener), then GLM's reply. The same tokens
carry different latents in a different context, so translators for live taps are fitted on rows from this layout.

Two modes:
  --items    test items {id, messages, memory, reply}: saves the tail and reply rows, the tail's length and the decoded
             text of both, so a receiver can be tested offline on the block the live connector sends.
  --triples  training passages, each placed as GLM's reply after a sampled instruction, with another passage in the
             memory: saves the reply rows and the reply tokens' character offsets. Only ASCII passages are kept, where
             byte-level tokens map one to one onto characters.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
from glm53_handoff import read_latents

SYSTEM = ("You are linked to another AI model through a shared memory that fills while you work. @@DRIFT@@ What your partner knows and "
          "writes arrives in that memory, never in this chat. Use it as your own recollection.")      # the loop's GLM system prompt
INSTRUCTIONS = ("Write the document.", "Write it up for the team, in plain prose.", "Draft the notice now.",
                "Put this into a short report: {fact}", "Notes you have: {fact} Write the document these notes are for.",
                "{fact} Write a few paragraphs about this.")
parser = argparse.ArgumentParser()
parser.add_argument("--items", type=Path, help="test items: json list of {id, messages, memory, reply}")
parser.add_argument("--triples", type=Path, help="training triples: passages by pid")
parser.add_argument("--split", choices=["train", "val"])
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--reserve", type=int, default=0, help="pad the memory with placeholder tokens to this many rows, as the live reserve holds")
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--seed", type=int, default=0)
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


def split(messages: list[dict]) -> tuple[list[int], list[int]]:
    """The chat template's tokens before and after @@DRIFT@@, as the live session splits them."""
    tokenized = post("/tokenize", {"model": args.model, "messages": messages, "add_generation_prompt": True, "return_token_strs": True,
                                   "chat_template_kwargs": {"enable_thinking": False}})
    ids, text, spans = tokenized["tokens"], "", []
    for piece in tokenized["token_strs"]:
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    a = text.index("@@DRIFT@@")
    hit = [i for i, (x, y) in enumerate(spans) if x < a + len("@@DRIFT@@") and y > a]
    return ids[:hit[0]], ids[hit[-1] + 1:]


def tokenize(text: str) -> tuple[list[int], list[tuple[int, int]], bool]:
    """Token ids, their character offsets, and whether the offsets are exact."""
    tokenized = post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False, "return_token_strs": True})
    offsets, at = [], 0
    for piece in tokenized["token_strs"]:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    return tokenized["tokens"], offsets, at == len(text) and text.isascii()


def latents_of(prompt: list[int], first: int, count: int) -> dict:
    name = f"blk-{uuid.uuid4().hex[:10]}"
    post("/v1/completions", {"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                             "cache_salt": f"drift:{uuid.uuid4().hex}", "kv_transfer_params": {"glm53_handoff": True, "handoff_id": name}})
    deadline = time.monotonic() + 60
    while not (ROOT / name / "rank0.ready").exists():
        if time.monotonic() > deadline:
            raise SystemExit("no export")
        time.sleep(0.05)
    latents = read_latents(export_bytes(name))
    subprocess.run(["rm", "-rf", str(ROOT / name)], check=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, f"rm -rf {ROOT / name}"], check=True, timeout=60)
    return {f"l{layer}": value[first:first + count].astype(np.float16) for layer, value in latents.items()}


def memory_rows(text: str) -> list[int]:
    ids = tokenize(text)[0]
    if args.reserve and len(ids) > args.reserve:
        raise SystemExit("the memory is longer than the reserve")
    return ids + [args.placeholder_id] * max(0, args.reserve - len(ids))


def first_sentences(text: str, count: int = 2) -> str:
    parts = text.split(". ")
    return ". ".join(parts[:count]).rstrip(".") + "."


started, done = time.time(), {p.stem for p in args.out.glob("*.npz")}
if args.items:
    texts = {}
    for item in json.loads(args.items.read_text()):
        head, tail = split(item["messages"])
        memory, reply = memory_rows(item["memory"]), tokenize(item["reply"])[0]
        rows = latents_of(head + memory + tail + reply, len(head) + len(memory), len(tail) + len(reply))
        np.savez(args.out / f"{item['id']}.npz", tail=np.int32(len(tail)), **rows)
        texts[item["id"]] = {"tail": post("/detokenize", {"model": args.model, "tokens": tail})["prompt"],
                             "reply": post("/detokenize", {"model": args.model, "tokens": reply})["prompt"], "tail_rows": len(tail), "reply_rows": len(reply)}
    (args.out / "texts.json").write_text(json.dumps(texts, indent=1) + "\n")
    print(json.dumps({"written": len(texts), "seconds": round(time.time() - started, 1)}))
    raise SystemExit
passages = {}
for triple in json.loads(args.triples.read_text())[args.split]:
    passages.setdefault(triple["pid"], triple["passage"])
pool, chooser, written, skipped = list(passages.values()), random.Random(args.seed), 0, 0
for pid, text in list(passages.items())[: args.limit]:
    other = chooser.choice([p for p in pool if p != text])
    instruction = chooser.choice(INSTRUCTIONS).format(fact=first_sentences(chooser.choice(pool), 1))
    if pid in done:
        continue
    ids, offsets, aligned = tokenize(text)
    if not aligned:
        skipped += 1
        continue
    head, tail = split([{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}])
    memory = memory_rows(first_sentences(other))
    rows = latents_of(head + memory + tail + ids, len(head) + len(memory) + len(tail), len(ids))
    np.savez(args.out / f"{pid}.npz", offsets=np.asarray(offsets, np.int32), aligned=np.bool_(True), **rows)
    written += 1
    if written % 50 == 0:
        print(json.dumps({"done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"written": written, "skipped": skipped, "seconds": round(time.time() - started, 1)}))
