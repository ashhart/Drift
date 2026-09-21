"""Bounded GLM live session for the reverse MCDMA test (Spark head, stdlib only; the API key comes from the environment and
is never printed). The reserved span stands where @@DRIFT@@ is in the system message. The span is padded so the prompt
length is a multiple of 64 and at least 64 own tokens follow it: on this build the first prefill step ends 64 tokens before
the end, the connector writes the span right after that step, and the LAST 64 own tokens (the question) are then computed
against the written memory. Prints {"ready"}, waits for one line on stdin, runs the request, prints the answer and timings."""
import argparse, json, os, shutil, sys, time, urllib.request, uuid
from pathlib import Path
from glm_completion_sse import CompletionStream
parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True)
parser.add_argument("--messages", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True, help="rows the publication will write; the span is at least this long")
parser.add_argument("--max-new", type=int, default=64)
parser.add_argument("--tap", action="store_true", help="continuous mode: export decode-time taps, no tail alignment, per-chunk timestamps from this host's clock")
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
for kind in ("tp-live-in", "tp-live-out"):
    if (ROOT / kind / args.session).exists():
        raise SystemExit("session folders already exist; use a fresh session name")
    (ROOT / kind / args.session).mkdir(parents=True, mode=0o700)


def post(path, body):
    return urllib.request.urlopen(urllib.request.Request(args.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}), timeout=600)


with post("/tokenize", {"model": args.model, "messages": json.loads(args.messages.read_text()), "add_generation_prompt": True, "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False}}) as resp:
    tokenized = json.loads(resp.read())
ids, strs = tokenized["tokens"], tokenized["token_strs"]
text, spans = "", []
for piece in strs:
    spans.append((len(text), len(text) + len(piece))); text += piece
a = text.index("@@DRIFT@@"); b = a + len("@@DRIFT@@")
hit = [i for i, (x, y) in enumerate(spans) if x < b and y > a]
head, tail = ids[:hit[0]], ids[hit[-1] + 1:]
if len(tail) < 64 and not args.tap:
    raise SystemExit(f"only {len(tail)} own tokens follow the span; at least 64 are needed so the question is computed after the write")
reserve = -(-args.rows // 64) * 64
reserve += (-(len(head) + reserve + len(tail))) % 64                            # total length a multiple of 64
prompt = head + [args.placeholder_id] * reserve + tail
body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_new, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "cache_salt": f"drift:{uuid.uuid4().hex}", "vllm_xargs": {"skip_writing_prefix_cache": 1},
        "kv_transfer_params": {"drift_session": args.session, "drift_reserve": reserve, "drift_reserve_start": len(head), "drift_tap": bool(args.tap)}}
print(json.dumps({"ready": True, "prompt_tokens": len(prompt), "span_start": len(head), "reserve": reserve, "own_tokens_after_span": len(tail)}), flush=True)
if sys.stdin.readline().strip() != "go":                                        # a closed control channel must NOT start the request
    for kind in ("tp-live-in", "tp-live-out"):
        shutil.rmtree(ROOT / kind / args.session, ignore_errors=True)
    raise SystemExit("no go: session abandoned before any request was sent")
t0, first, pieces, chunks, wall0 = time.time(), None, [], [], time.time_ns()
completion = CompletionStream()
with post("/v1/completions", body) as resp:
    for raw in resp:
        piece = completion.accept(raw)
        if completion.done:
            break
        if piece:
            if first is None:
                first = time.time() - t0
            pieces.append(piece); chunks.append([time.time_ns(), len(piece)])
terminal = completion.finish(prompt_tokens=len(prompt), max_tokens=args.max_new)
if args.tap:
    print(json.dumps({"stream": {"request_wall_ns": wall0, "chunks": chunks}}), flush=True)
finished = ROOT / "tp-live-out" / args.session / "finished"
print(json.dumps({"text": "".join(pieces), "first_token_s": round(first, 4) if first is not None else None, "total_s": round(time.time() - t0, 4), "connector_finished": json.loads(finished.read_text()) if finished.exists() else None, **terminal}), flush=True)
