"""Qualify the Drift Qwen3.8 connector on the live Sparks. Run on the head Spark, next to qwen38_export.py and qwen38_pages.py.

Every prompt for a passage shares one layout: an opening, a span of T tokens at [S, S + T), newline filler up to own start
O, then the question. native puts the passage in the span. An export request reads the native span's pages back through
the owner's handoff flag. inject puts placeholders in the span and has the connector write the exported rows over them.
no_memory keeps the placeholders. The first inject is also exported, to check that the write landed bit for bit.

Gate, fixed before the first run. WRITE passes when that export equals the injected rows exactly on every layer and rank.
RECALL passes when inject exact match is at least 0.8 of native and no_memory is at most 0.2. Recurrent layers see
placeholders in both inject and no_memory, so RECALL measures what the attention path alone carries.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
from qwen38_export import read_blob, span_rows, span_selector

parser = argparse.ArgumentParser()
parser.add_argument("--api", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="qwen3.8-flash-next")
parser.add_argument("--jobs", type=Path, required=True, help="JSON list of {text, questions: [{question, answer}]}")
parser.add_argument("--own-start", type=int, default=8192, help="the engine's first prefill step ends here")
parser.add_argument("--handoff", type=Path, default=Path("/dev/shm/qwen38-drift"))
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host")
parser.add_argument("--max-new", type=int, default=48)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("--out already exists")
RUN = f"drq-{uuid.uuid4().hex[:8]}"
INJECT = args.handoff / "drift-inject"


def post(path: str, body: dict, timeout: float = 900) -> dict:
    request = urllib.request.Request(args.api + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def tokens(text: str) -> list[int]:
    return post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False})["tokens"]


def complete(prompt: list[int], max_tokens: int, params: dict | None = None) -> tuple[str, float]:
    body = {"model": args.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "cache_salt": uuid.uuid4().hex}
    if params:
        body["kv_transfer_params"] = params
    started = time.perf_counter()
    text = post("/v1/completions", body)["choices"][0]["text"]
    return text, round(time.perf_counter() - started, 3)


def worker(command: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, command], input=data, capture_output=True, check=True).stdout


def wait_ready(handoff_id: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    local = args.handoff / handoff_id / "rank0.ready"
    while time.monotonic() < deadline:
        remote = worker(f"test -f {args.handoff / handoff_id / 'rank1.ready'} && echo yes || true").strip() == b"yes"
        if local.exists() and remote:
            return
        time.sleep(0.2)
    raise TimeoutError(f"export {handoff_id} not ready on both ranks")


def exported(handoff_id: str) -> list[tuple[dict, np.ndarray]]:
    wait_ready(handoff_id)
    local = (args.handoff / handoff_id / "rank0.bin").read_bytes()
    remote = worker(f"cat {args.handoff / handoff_id / 'rank1.bin'}")
    subprocess.run(["rm", "-rf", str(args.handoff / handoff_id)], check=True)
    worker(f"rm -rf {args.handoff / handoff_id}")
    return [read_blob(local), read_blob(remote)]


def publish(name: str, entries: dict) -> None:
    INJECT.mkdir(parents=True, exist_ok=True)
    path = INJECT / f"{name}.npz"
    np.savez(path, **entries)
    worker(f"mkdir -p {INJECT} && cat > {path}", path.read_bytes())


def receipts(name: str) -> dict:
    status = {}
    for rank, listing in ((0, "\n".join(p.name for p in INJECT.glob(f"{name}.*"))), (1, worker(f"ls {INJECT} | grep '^{name}\\.' || true").decode())):
        names = set(listing.split())
        status[rank] = "done" if f"{name}.rank{rank}.done" in names else ("error" if f"{name}.rank{rank}.error" in names else "missing")
        if f"{name}.preempted" in names:
            status[rank] = "preempted"
    return status


FILL = tokens("\n")
if len(FILL) != 1:
    raise SystemExit("the newline filler must be one token")
opening = tokens("<|im_start|>user\nRead this document, then answer the question after it.\n\n")
opening += FILL * ((-len(opening)) % 4)
S = len(opening)
jobs = json.loads(args.jobs.read_text())
rows, write_check, started = [], None, time.time()
for n, job in enumerate(jobs):
    passage = tokens(job["text"])
    T = len(passage) + (-len(passage)) % 4
    if S + T >= args.own_start:
        raise SystemExit(f"passage {n}: span {S}..{S + T} does not fit before own start {args.own_start}")
    native_span, blank_span = passage + FILL * (T - len(passage)), FILL * T
    gap = FILL * (args.own_start - S - T)
    handoff_id = f"{RUN}-x{n}"
    complete(opening + native_span + FILL * 4, 1, {"qwen38_handoff": True, "handoff_id": handoff_id})
    blobs = exported(handoff_id)
    kv, selector = span_rows(blobs, S, T), span_selector(blobs[0], S, T)
    entries = {**{f"kr{l}": k for l, (k, _) in kv.items()}, **{f"v{l}": v for l, (_, v) in kv.items()}, **{f"c{l}": c for l, c in selector.items()}}
    for m, q in enumerate(job["questions"]):
        own = tokens(f"\n\nQuestion: {q['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        name = f"{RUN}-{n}-{m}"
        publish(name, entries)
        params = {"drift_inject": name, "drift_tokens": T, "drift_span_start": S, "drift_own_start": args.own_start}
        if write_check is None:                                     # once: the inject request itself exported after the write
            check_id = f"{RUN}-v"
            complete(opening + blank_span + gap + own, 1, {**params, "qwen38_handoff": True, "handoff_id": check_id})
            after = exported(check_id)
            got, got_selector = span_rows(after, S, T), span_selector(after[0], S, T)
            write_check = {"receipts": receipts(name),
                           "keys_exact": all(np.array_equal(got[l][0], entries[f"kr{l}"]) for l in kv),
                           "values_exact": all(np.array_equal(got[l][1], entries[f"v{l}"]) for l in kv),
                           "selector_exact": all(np.array_equal(got_selector[l], entries[f"c{l}"]) for l in selector),
                           "layers": len(kv)}
            name = f"{RUN}-{n}-{m}b"
            publish(name, entries)
            params["drift_inject"] = name
        native = complete(opening + native_span + gap + own, args.max_new)
        inject = complete(opening + blank_span + gap + own, args.max_new, params)
        inject_receipts = receipts(name)
        blank = complete(opening + blank_span + gap + own, args.max_new)
        hit = lambda text: q["answer"].lower() in text.lower()
        rows.append({"passage": n, "question": m, "answer": q["answer"], "span": [S, S + T], "passage_tokens": len(passage),
                     "native": {"text": native[0], "correct": hit(native[0]), "s": native[1]},
                     "inject": {"text": inject[0], "correct": hit(inject[0]), "s": inject[1], "receipts": inject_receipts},
                     "no_memory": {"text": blank[0], "correct": hit(blank[0]), "s": blank[1]}})
        print(n, m, "native", rows[-1]["native"]["correct"], "inject", rows[-1]["inject"]["correct"], inject_receipts,
              "no_memory", rows[-1]["no_memory"]["correct"], flush=True)
valid = [r for r in rows if set(r["inject"]["receipts"].values()) == {"done"}]
rate = lambda arm: round(sum(r[arm]["correct"] for r in valid) / max(1, len(valid)), 4)
em = {arm: rate(arm) for arm in ("native", "inject", "no_memory")}
write_pass = bool(write_check and set(write_check["receipts"].values()) == {"done"} and write_check["keys_exact"]
                  and write_check["values_exact"] and write_check["selector_exact"])
recall_pass = len(valid) == len(rows) and em["inject"] >= 0.8 * em["native"] and em["no_memory"] <= 0.2
report = {"run": RUN, "own_start": args.own_start, "span_start": S, "questions": len(rows), "valid": len(valid), "exact_match": em,
          "write_check": write_check, "gate": {"WRITE": "PASSED" if write_pass else "FAILED", "RECALL": "PASSED" if recall_pass else "FAILED"},
          "seconds": round(time.time() - started, 1), "rows": rows}
args.out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("run", "questions", "valid", "exact_match", "write_check", "gate", "seconds")}))
