"""Capture GLM teacher data for distilling the reverse translator. Run on the head Spark through spark_run.sh.

Each passage becomes one prompt shaped like the live reverse loop: the chat template's tokens before @@DRIFT@@, the passage's
own GLM tokens where the memory span would stand, the tokens after @@DRIFT@@, then the correct answers to the passage's
questions, forced. One request exports GLM's latent cache for the whole prompt and captures GLM's latent-space queries at
the first two tokens of every answer, on both ranks. Each passage is saved as one npz of float16 latents and queries.
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
        "you were never shown it as text. @@DRIFT@@ Answer the user's questions from what you recall from that shared memory. "
        "Give each value directly, one per line, numbered like the questions.")
parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path, required=True)
parser.add_argument("--split", choices=["train", "val"], required=True)
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host")
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
args.out.mkdir(parents=True, exist_ok=True)


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def tokens(text: str) -> list[int]:
    return post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False})["tokens"]


def split_template(questions: list[str]) -> tuple[list[int], list[int]]:
    user = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    tokenized = post("/tokenize", {"model": args.model, "messages": [{"role": "system", "content": LINK}, {"role": "user", "content": user}],
                                   "add_generation_prompt": True, "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False}})
    ids, text, spans = tokenized["tokens"], "", []
    for piece in tokenized["token_strs"]:
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    a = text.index("@@DRIFT@@")
    hit = [i for i, (x, y) in enumerate(spans) if x < a + len("@@DRIFT@@") and y > a]
    return ids[:hit[0]], ids[hit[-1] + 1:]


def worker(command: str) -> bytes:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, command], capture_output=True, check=True, timeout=120).stdout


def wait(paths_local: list[Path], paths_remote: list[Path], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remote = worker(" && ".join(f"test -e {p}" for p in paths_remote) + " && echo yes || true").strip() == b"yes"
        if remote and all(p.exists() for p in paths_local):
            return
        time.sleep(0.2)
    raise TimeoutError("export or capture did not finish on both ranks")


passages = {}
for triple in json.loads(args.triples.read_text())[args.split]:
    passages.setdefault(triple["pid"], {"passage": triple["passage"], "qa": []})["qa"].append((triple["question"], triple["answer"], triple["kind"]))
done = {p.stem for p in args.out.glob("*.npz")}
started, written = time.time(), 0
for pid, item in list(passages.items())[: args.limit]:
    if pid in done:
        continue
    qa = item["qa"][:8]
    head, tail = split_template([q for q, _, _ in qa])
    passage = tokens(item["passage"])
    answer_ids, rows, row_question = [], [], []
    base = len(head) + len(passage) + len(tail)
    for i, (_, answer, _) in enumerate(qa):
        answer_ids += tokens(f"{i + 1}. ")
        value = tokens(answer)
        rows += [base + len(answer_ids) + k for k in range(min(2, len(value)))]
        row_question += [i] * min(2, len(value))
        answer_ids += value + tokens("\n")
    prompt = head + passage + tail + answer_ids
    name = f"cap-{uuid.uuid4().hex[:12]}"
    post("/v1/completions", {"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "cache_salt": f"drift:{uuid.uuid4().hex}",
                             "kv_transfer_params": {"glm53_handoff": True, "handoff_id": name, "drift_capture": name, "drift_capture_rows": rows}})
    capture = ROOT / "drift-capture"
    wait([ROOT / name / "rank0.ready", capture / f"{name}.rank0.done"], [capture / f"{name}.rank1.done"])
    latents = read_latents(ROOT / name / "rank0.bin")
    local = np.load(capture / f"{name}.rank0.npz")
    remote_path = args.out / f".{name}.rank1.npz"
    remote_path.write_bytes(worker(f"cat {capture / f'{name}.rank1.npz'}"))
    remote = np.load(remote_path)
    layers = sorted(int(k[1:]) for k in local.files if k.startswith("l"))
    if int(local["head_offset"]) != 0 or local["positions"].tolist() != rows or remote["positions"].tolist() != rows:
        raise SystemExit(f"{pid}: capture rows or head order do not match the request")
    queries = {f"q{l}": np.concatenate((local[f"l{l}"], remote[f"l{l}"]), axis=1) for l in layers}
    np.savez(args.out / f"{pid}.npz", span=np.asarray([len(head), len(head) + len(passage)], dtype=np.int32), rows=np.asarray(rows, dtype=np.int32), row_question=np.asarray(row_question, dtype=np.int32),
             scale=local["scale"], prompt_tokens=np.int32(len(prompt)), **queries,
             **{f"c{l}": latents[l][: len(prompt)].astype(np.float16) for l in layers})
    remote_path.unlink()
    subprocess.run(["rm", "-rf", str(ROOT / name)], check=True)
    for suffix in ("npz", "done"):
        (capture / f"{name}.rank0.{suffix}").unlink(missing_ok=True)
    worker(f"rm -rf {ROOT / name} {capture / name}.rank1.npz {capture / name}.rank1.done")
    written += 1
    print(json.dumps({"pid": pid, "prompt_tokens": len(prompt), "rows": len(rows), "done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"split": args.split, "written": written, "seconds": round(time.time() - started, 1)}))
