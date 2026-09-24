"""Qualify the raw-row Drift connector on DeepSeek V4 on the live Sparks. Run on the head Spark, next to raw_rows.py.

Every prompt for a passage shares one layout: an opening, a span of T tokens at [S, S + T) with S and T multiples of 128,
newline filler up to own start O, then the question, as a plain completion. native puts the passage in the span. A tap
request reads the native span's compressed entries back from every rank. inject puts placeholders in the span and has
the connector write rank 0's tapped rows over them. no_memory keeps the placeholders. The first inject also taps itself.

Gate, fixed before the first run. WRITE passes when that tap equals the injected rows exactly on every tensor and rank,
and both ranks tapped identical rows from the native span. RECALL passes when inject exact match is at least 0.8 of
native and no_memory is at most 0.2. Sliding-window and compressor state see placeholders, so RECALL measures what
the compressed caches alone carry.

With --translated, each job naming a pid also runs one arm per translated_<sender>_<pid>.npz there (hub_fit_dsv4.py):
the placeholder span is tapped once, the translated content replaces its entries' content (drift/translate/dsv4_member.py),
and the result is injected. Translated entries are whole pages, so --align 256 is needed.
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
from raw_rows import load, save as save_rows
from drift.translate import dsv4_member

parser = argparse.ArgumentParser()
parser.add_argument("--api", default="http://127.0.0.1:8888")
parser.add_argument("--model", help="served model name; default: the only one /v1/models lists")
parser.add_argument("--jobs", type=Path, required=True, help="JSON list of {text, questions: [{question, answer}]}")
parser.add_argument("--own-start", type=int, default=8192, help="the engine's first prefill step ends here")
parser.add_argument("--path", type=Path, default=Path("/dev/shm/drift-rows"))
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host")
parser.add_argument("--max-new", type=int, default=48)
parser.add_argument("--align", type=int, default=128, help="span alignment; 256 keeps whole cache pages for translated arms")
parser.add_argument("--translated", type=Path, help="folder of translated_<sender>_<pid>.npz for jobs that name a pid")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("--out already exists")
KEY = os.environ.get("DRIFT_API_KEY", "")                           # read from the host's environment, never stored
RUN, ALIGN = f"drd-{uuid.uuid4().hex[:8]}", args.align


def post(path: str, body: dict | None = None, timeout: float = 900) -> dict:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})}
    request = urllib.request.Request(args.api + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


MODEL = args.model or post("/v1/models")["data"][0]["id"]


def tokens(text: str) -> list[int]:
    return post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]


def complete(prompt: list[int], max_tokens: int, params: dict | None = None) -> tuple[str, float]:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "cache_salt": uuid.uuid4().hex}
    if params:
        body["kv_transfer_params"] = params
    started = time.perf_counter()
    text = post("/v1/completions", body)["choices"][0]["text"]
    return text, round(time.perf_counter() - started, 3)


def worker(command: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, command], input=data, capture_output=True, check=True).stdout


def marks(name: str) -> dict:
    local = {p.name.removeprefix(f"{name}.") for p in (args.path / "drift-marks").glob(f"{name}.*")}
    remote = set(worker(f"ls {args.path / 'drift-marks'} 2>/dev/null | grep '^{name}\\.' || true").decode().split())
    return {kind: {rank: f"{kind}.rank{rank}.done" in (local if rank == 0 else {n.removeprefix(f'{name}.') for n in remote})
                   for rank in (0, 1)} for kind in ("inject", "tap")} | {"preempted": f"{name}.preempted" in local}


def tapped(name: str, timeout: float = 60) -> list[dict]:
    deadline = time.monotonic() + timeout
    local = args.path / "drift-tap" / f"{name}.rank0.npz"
    remote = args.path / "drift-tap" / f"{name}.rank1.npz"
    while not (local.exists() and worker(f"test -f {remote} && echo yes || true").strip() == b"yes"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"tap {name} not written on both ranks")
        time.sleep(0.2)
    copy = args.out.parent / f".{name}.rank1.npz"
    copy.write_bytes(worker(f"cat {remote}"))
    rows = [load(local), load(copy)]
    copy.unlink()
    return rows


def publish(name: str, source: Path) -> None:
    target = args.path / "drift-inject" / f"{name}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    worker(f"mkdir -p {target.parent} && cat > {target}", target.read_bytes())


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k][0] == b[k][0] and np.array_equal(a[k][1], b[k][1]) for k in a)


FILL = tokens("\n")
if len(FILL) != 1:
    raise SystemExit("the newline filler must be one token")
opening = tokens("Read this document, then answer the question after it.\n\nDocument:\n")
opening += FILL * ((-len(opening)) % ALIGN)
S = len(opening)
jobs = json.loads(args.jobs.read_text())
rows, write_check, started = [], None, time.time()
for n, job in enumerate(jobs):
    passage = tokens(job["text"])
    T = len(passage) + (-len(passage)) % ALIGN
    if S + T >= args.own_start:
        raise SystemExit(f"passage {n}: span {S}..{S + T} does not fit before own start {args.own_start}")
    native_span, blank_span = passage + FILL * (T - len(passage)), FILL * T
    gap = FILL * (args.own_start - S - T)
    span = {"drift_tokens": T, "drift_span_start": S}
    tap_name = f"{RUN}-t{n}"
    complete(opening + native_span + FILL * 4, 1, {"drift_tap": tap_name, **span})
    native_rows = tapped(tap_name)
    replicated = same(native_rows[0], native_rows[1])
    senders = {}
    if args.translated and job.get("pid"):
        files = sorted(args.translated.glob(f"translated_*_{job['pid']}.npz"))
        if files:
            blank_name = f"{RUN}-b{n}"
            complete(opening + blank_span + FILL * 4, 1, {"drift_tap": blank_name, **span})
            blank_rows = tapped(blank_name)[0]
            for path in files:
                t = np.load(path)
                senders[path.stem.split("_")[1]] = dsv4_member.inject_rows(blank_rows, t["mla4"], t["index4"], t["kept4"], t["mla128"], t["kept128"])
    for m, q in enumerate(job["questions"]):
        own = tokens(f"\n\nQuestion: {q['question']}\nAnswer:")
        name = f"{RUN}-{n}-{m}"
        publish(name, args.path / "drift-tap" / f"{tap_name}.rank0.npz")
        params = {"drift_inject": name, **span, "drift_own_start": args.own_start}
        if write_check is None:                                     # once: the inject request also taps its own span
            complete(opening + blank_span + gap + own, 1, {**params, "drift_tap": name})
            after = tapped(name)
            write_check = {"marks": marks(name), "ranks_replicated_on_native": replicated,
                           "rank0_exact": same(after[0], native_rows[0]), "rank1_exact": same(after[1], native_rows[0]),
                           "tensors": len(native_rows[0])}
            name = f"{RUN}-{n}-{m}b"
            publish(name, args.path / "drift-tap" / f"{tap_name}.rank0.npz")
            params["drift_inject"] = name
        native = complete(opening + native_span + gap + own, args.max_new)
        inject = complete(opening + blank_span + gap + own, args.max_new, params)
        inject_marks = marks(name)
        blank = complete(opening + blank_span + gap + own, args.max_new)
        hit = lambda text: q["answer"].lower() in text.lower()
        translated = {}
        for sender, sender_rows in senders.items():
            t_name = f"{RUN}-{n}-{m}-{sender}"
            source = args.out.parent / f".{t_name}.npz"
            save_rows(source, sender_rows)
            publish(t_name, source)
            source.unlink()
            answer = complete(opening + blank_span + gap + own, args.max_new, {**params, "drift_inject": t_name})
            translated[f"t_{sender}"] = {"text": answer[0], "correct": hit(answer[0]), "s": answer[1], "marks": marks(t_name)}
        rows.append({"passage": n, "question": m, "answer": q["answer"], "span": [S, S + T], "passage_tokens": len(passage),
                     "native_ranks_replicated": replicated,
                     "native": {"text": native[0], "correct": hit(native[0]), "s": native[1]},
                     "inject": {"text": inject[0], "correct": hit(inject[0]), "s": inject[1], "marks": inject_marks},
                     "no_memory": {"text": blank[0], "correct": hit(blank[0]), "s": blank[1]}, **translated})
        print(n, m, "native", rows[-1]["native"]["correct"], "inject", rows[-1]["inject"]["correct"], inject_marks["inject"],
              "no_memory", rows[-1]["no_memory"]["correct"], *[f"{k} {v['correct']}" for k, v in translated.items()], flush=True)
valid = [r for r in rows if all(r["inject"]["marks"]["inject"].values()) and not r["inject"]["marks"]["preempted"]]
rate = lambda arm: round(sum(r[arm]["correct"] for r in valid) / max(1, len(valid)), 4)
em = {arm: rate(arm) for arm in ("native", "inject", "no_memory")}
for arm in sorted({k for r in valid for k in r if k.startswith("t_")}):
    scored = [r for r in valid if arm in r and all(r[arm]["marks"]["inject"].values()) and not r[arm]["marks"]["preempted"]]
    em[arm] = {"exact_match": round(sum(r[arm]["correct"] for r in scored) / max(1, len(scored)), 4), "correct": sum(r[arm]["correct"] for r in scored),
               "questions": len(scored)}
write_pass = bool(write_check and all(write_check["marks"]["inject"].values()) and all(write_check["marks"]["tap"].values())
                  and write_check["ranks_replicated_on_native"] and write_check["rank0_exact"] and write_check["rank1_exact"])
recall_pass = len(valid) == len(rows) and em["inject"] >= 0.8 * em["native"] and em["no_memory"] <= 0.2
report = {"run": RUN, "model": MODEL, "own_start": args.own_start, "span_start": S, "questions": len(rows), "valid": len(valid),
          "exact_match": em, "write_check": write_check,
          "gate": {"WRITE": "PASSED" if write_pass else "FAILED", "RECALL": "PASSED" if recall_pass else "FAILED"},
          "seconds": round(time.time() - started, 1), "rows": rows}
args.out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("run", "questions", "valid", "exact_match", "write_check", "gate", "seconds")}))
