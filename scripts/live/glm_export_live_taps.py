"""Collect GLM's forward taps for training passages through the live connector itself, the rows a partner receives.

Runs on the head Spark through spark_run.sh, against the Drift server. Each passage becomes one causal live session, as
spark_glm_session.py builds it: the loop's system prompt with a reserve where @@DRIFT@@ stands, sized the same way, a
sampled user instruction, then the passage as GLM's reply, forced as prompt tokens. Before the request the session's
first publication is staged on both ranks: another passage's rows from Qwen, translated by translate_passages.py, in
three copies, as the loop publishes Qwen's prompt. The prefill stops at the reserve's end, the connector writes the
memory, and GLM reads its instruction and the passage against it while the connector taps every row after the reserve.
The passage's rows are kept with their tokens' character offsets. Only ASCII passages are kept, where byte-level tokens
map one to one onto characters.

With --items instead, each item names GLM's own messages, a memory file of translated rows (such as
studio_replay_memory.py rebuilds for a live run) and GLM's reply: the whole tapped block, GLM's prompt tail and reply, is
saved with the tail's length, to reproduce a live session's taps offline. With --state, the memory file's translated KDA
inputs (h{layer}, from translate_passages.py) also go to GLM, as in the reverse state gate: GLM's state after its head
alone is exported, and the connector advances it by those inputs at the reserve's end. With --generate N, GLM writes its
own reply of up to N tokens instead, as the live session does; the item then needs no reply, and the reply text is saved
with the taps.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import random
import shlex
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np

SYSTEM = ("You are linked to another AI model through a shared memory that fills while you work. @@DRIFT@@ What your partner knows and "
          "writes arrives in that memory, never in this chat. Use it as your own recollection.")      # the loop's GLM system prompt
INSTRUCTIONS = ("Write the document.", "Write it up for the team, in plain prose.", "Draft the notice now.",
                "Put this into a short report: {fact}", "Notes you have: {fact} Write the document these notes are for.",
                "{fact} Write a few paragraphs about this.")
parser = argparse.ArgumentParser()
parser.add_argument("--triples", type=Path)
parser.add_argument("--split", choices=["train", "val"])
parser.add_argument("--items", type=Path, help="json list of {id, messages, memory, reply}; memory names an npz of l{layer} rows")
parser.add_argument("--memory", type=Path, help="translated Qwen rows per passage, l{layer} arrays from translate_passages.py")
parser.add_argument("--memory-rows", type=int, default=112, help="rows of one memory passage, like the loop's Qwen prompt")
parser.add_argument("--copies", type=int, default=3)
parser.add_argument("--state", action="store_true", help="with --items: also send the memory's translated state")
parser.add_argument("--generate", type=int, help="with --items: GLM writes its own reply of up to this many tokens")
parser.add_argument("--rows", type=int, default=1536, help="the loop's --reserve: rows the session reserves before padding")
parser.add_argument("--limit", type=int, default=100000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", required=True, help="ssh target of the rank-1 host")
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
if bool(args.items) == bool(args.triples and args.split and args.memory):
    parser.error("give --items, or --triples with --split and --memory")
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
args.out.mkdir(parents=True, exist_ok=True)


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def worker(command: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", args.worker, command], input=data, capture_output=True, check=True, timeout=120).stdout


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


def stage(session: str, arrays: dict, name: str = "000000.npz") -> None:
    """One npz in the session's inbox on both hosts, renamed into place only when complete."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    folder = ROOT / "tp-live-in" / session
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f".{name}.tmp").write_bytes(buffer.getvalue())
    os.replace(folder / f".{name}.tmp", folder / name)
    remote = shlex.quote(str(folder))
    worker(f"mkdir -p {remote} && cat > {remote}/.{name}.tmp && mv {remote}/.{name}.tmp {remote}/{name}", buffer.getvalue())


def on_both(local: Path, remote: Path, timeout: float = 120) -> None:
    """Wait until rank 0's file is on this host and rank 1's on the worker."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if local.exists() and worker(f"test -e {shlex.quote(str(remote))} && echo yes || true").strip() == b"yes":
            return
        time.sleep(0.2)
    raise SystemExit(f"not on both ranks: {local.name}, {remote.name}")


def head_state(head: list[int]) -> str:
    """Export GLM's state after its head alone, and refuse unless both ranks hold it: a failed state write stops the engine."""
    name = f"head-{uuid.uuid4().hex[:10]}"
    post("/v1/completions", {"model": args.model, "prompt": head, "max_tokens": 1, "temperature": 0, "cache_salt": f"drift:{uuid.uuid4().hex}",
                             "vllm_xargs": {"skip_writing_prefix_cache": 1}, "kv_transfer_params": {"glm53_handoff": True, "handoff_id": name}})
    on_both(ROOT / name / "rank0.ready", ROOT / name / "rank1.ready")
    probe = ("import json, os, sys, time; r = json.load(open(sys.argv[1] + '/rank' + sys.argv[2] + '.ready')); "
             "ok = r.get('mode') == 'arena' or os.path.exists(sys.argv[1] + '/rank' + sys.argv[2] + '.bin'); "
             "print('ok' if ok and time.time() - float(r.get('created', 0)) < 90 else 'stale')")
    local = subprocess.run(["python3", "-c", probe, str(ROOT / name), "0"], capture_output=True, text=True, check=True).stdout.strip()
    remote = worker(f"python3 -c {shlex.quote(probe)} {shlex.quote(str(ROOT / name))} 1").decode().strip()
    if (local, remote) != ("ok", "ok"):
        raise SystemExit(f"head export {name} is not usable on both ranks: {local}, {remote}")
    return name


def taps(session: str, start: int, stop: int | None) -> dict[str, np.ndarray]:
    """The session's taps joined in order, after the connector's finished marker vouches for them; stop None takes its own."""
    folder = ROOT / "tp-live-out" / session
    deadline = time.monotonic() + 60
    while not (folder / "finished").exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{session}: no finished marker")
        time.sleep(0.05)
    outcome = json.loads((folder / "finished").read_text())
    stop = outcome.get("source_stop") if stop is None else stop
    if outcome.get("failed") or outcome.get("source_start") != start or outcome.get("source_stop") != stop:
        raise SystemExit(f"{session}: the connector finished with {outcome}")
    parts, cursor = [], start
    for index in range(outcome["tap_count"]):
        tap = np.load(folder / f"{index:06d}.npz")
        if int(tap["start"]) != cursor:
            raise SystemExit(f"{session}: taps are not contiguous")
        cursor = int(tap["stop"])
        parts.append({k: tap[k] for k in tap.files if k[0] == "l" and k[1:].isdigit()})
    if cursor != stop:
        raise SystemExit(f"{session}: taps end at {cursor}, not {stop}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def first_sentence(text: str) -> str:
    return text.split(". ")[0].rstrip(".") + "."


def session_rows(messages: list[dict], rows: dict, reply: list[int], inputs: dict | None = None, generate: int | None = None) -> tuple[dict, int, str]:
    """One causal live session with `rows` as its first publication: GLM's tapped rows after the reserve, the tail's length,
    and any text GLM wrote. With `inputs`, GLM's head state is advanced by them at the reserve's end, as the reverse state
    gate does. With `generate`, GLM writes up to that many tokens after its prompt instead of reading a forced reply."""
    head, tail = split(messages)
    reserve = -(-args.rows // 64) * 64
    reserve += (-(len(head) + reserve + len(tail))) % 64                            # spark_glm_session.py's sizing
    session, blob = f"lt-{uuid.uuid4().hex[:12]}", None
    prompt = head + [args.placeholder_id] * reserve + tail + reply
    params = {"drift_session": session, "drift_reserve": reserve, "drift_reserve_start": len(head), "drift_prefill_boundary": len(head) + reserve, "drift_tap": True}
    try:
        stage(session, rows)
        if inputs is not None:
            stage(session, inputs, "state.npz")
            blob = head_state(head)
            params.update(drift_state_blob=blob, drift_state_rows=True)
        answer = post("/v1/completions", {"model": args.model, "prompt": prompt, "max_tokens": generate or 1, "temperature": 0,
                                          "cache_salt": f"drift:{uuid.uuid4().hex}", "vllm_xargs": {"skip_writing_prefix_cache": 1}, "kv_transfer_params": params})
        tapped = taps(session, len(head) + reserve, None if generate else len(prompt))
        if inputs is not None:
            on_both(*(ROOT / "tp-live-out" / session / f"state.rank{r}.json" for r in (0, 1)), timeout=30)
        return tapped, len(tail), answer["choices"][0]["text"] if generate else ""
    finally:
        paths = " ".join(shlex.quote(str(ROOT / kind / session)) for kind in ("tp-live-in", "tp-live-out"))
        paths += f" {shlex.quote(str(ROOT / blob))}" if blob else ""
        subprocess.run(f"rm -rf {paths}", shell=True, check=True)
        worker(f"rm -rf {paths}")


def tiled(path: Path, count: int | None) -> dict:
    memory = np.load(path)
    return {k: np.tile(memory[k][:count].astype(np.float32), (args.copies, 1)) for k in memory.files if k[0] == "l" and k[1:].isdigit()}


if args.items:
    started = time.time()
    texts = {}
    for item in json.loads(args.items.read_text()):
        reply = [] if args.generate else post("/tokenize", {"model": args.model, "prompt": item["reply"], "add_special_tokens": False})["tokens"]
        memory = np.load(item["memory"])
        inputs = {k: memory[k] for k in memory.files if k[0] == "h" and k[1:].isdigit()} if args.state else None
        if args.state and not inputs:
            raise SystemExit(f"{item['id']}: --state needs h{{layer}} inputs in its memory file")
        tapped, tail, text = session_rows(item.get("messages") or item["glm"], tiled(Path(item["memory"]), None), reply, inputs, args.generate)
        np.savez(args.out / f"{item['id']}.npz", tail=np.int32(tail), **{k: v.astype(np.float16) for k, v in tapped.items()})
        texts[item["id"]] = text
    if args.generate:
        (args.out / "texts.json").write_text(json.dumps(texts, indent=1) + "\n")
    print(json.dumps({"written": len(list(args.out.glob("*.npz"))), "seconds": round(time.time() - started, 1)}))
    raise SystemExit
passages = {}
for triple in json.loads(args.triples.read_text())[args.split]:
    passages.setdefault(triple["pid"], triple["passage"])
pool = [pid for pid in passages if (args.memory / f"{pid}.npz").exists()]
chooser, done, started, written, skipped = random.Random(args.seed), {p.stem for p in args.out.glob("*.npz")}, time.time(), 0, 0
for pid, text in list(passages.items())[: args.limit]:
    other = chooser.choice([p for p in pool if p != pid])
    instruction = chooser.choice(INSTRUCTIONS).format(fact=first_sentence(passages[chooser.choice(pool)]))
    if pid in done:
        continue
    tokenized = post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False, "return_token_strs": True})
    ids, offsets, at = tokenized["tokens"], [], 0
    for piece in tokenized["token_strs"]:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    if at != len(text) or not text.isascii():
        skipped += 1
        continue
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
    tapped, tail, _ = session_rows(messages, tiled(args.memory / f"{other}.npz", args.memory_rows), ids)
    np.savez(args.out / f"{pid}.npz", offsets=np.asarray(offsets, np.int32), aligned=np.bool_(True),
             **{k: v[tail:].astype(np.float16) for k, v in tapped.items()})
    written += 1
    if written % 50 == 0:
        print(json.dumps({"done": written, "seconds": round(time.time() - started, 1)}), flush=True)
print(json.dumps({"written": written, "skipped": skipped, "seconds": round(time.time() - started, 1)}))
