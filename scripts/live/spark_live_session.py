"""Run one GLM session with optional controller-supervised cancellation."""
import argparse, json, math, os, shutil, sys, time, urllib.request, uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True)
parser.add_argument("--messages", type=Path, required=True, help="json list of chat messages")
parser.add_argument("--reserve", type=int, default=1024)
parser.add_argument("--max-new", type=int, default=300)
parser.add_argument("--max-prompt-tokens", type=int, help="reject an oversized prompt before completion submission")
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
parser.add_argument("--control-stdin", action="store_true", help="opt in to typed abort or EOF cancellation over stdin")
parser.add_argument("--no-link", action="store_true", help="keep the same placeholder prompt but disable live cache reads and exports")
parser.add_argument("--no-tap", action="store_true", help="disable outbound exports for a one-way linked request")
parser.add_argument("--wait-publication", action="store_true", help="wait for typed release before the completion POST")
parser.add_argument("--ready-timeout", type=float, default=180, help="maximum seconds to wait for publication release")
args = parser.parse_args()
if not math.isfinite(args.ready_timeout) or not 0 < args.ready_timeout <= 180:
    parser.error("ready-timeout must be positive and at most 180 seconds")
if args.max_prompt_tokens is not None and args.max_prompt_tokens <= 0:
    parser.error("max-prompt-tokens must be positive")
if args.wait_publication and args.no_link:
    parser.error("publication release is only valid for linked requests")
if args.control_stdin:
    try:
        from live_session_control import SessionCancelled, SessionControlError, supervise
    except ImportError:
        from drift.serving.live_session_control import SessionCancelled, SessionControlError, supervise
    command = [sys.executable, str(Path(__file__).resolve()), *(arg for arg in sys.argv[1:] if arg != "--control-stdin")]
    try:
        supervise(command, sys.stdin.fileno(), lambda event: print(json.dumps(event), flush=True), allow_release=args.wait_publication)
    except SessionCancelled:
        print(json.dumps({"cancelled": True}), flush=True)
        raise SystemExit(2)
    except SessionControlError:
        print(json.dumps({"failed": True}), flush=True)
        raise SystemExit(3)
    raise SystemExit(0)
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
if args.wait_publication:
    from drift.serving.live_start_gate import prepare_fresh, wait_release
    prepare_fresh(ROOT, args.session)
elif not args.no_link:
    for folder in (ROOT / "tp-live-in" / args.session, ROOT / "tp-live-out" / args.session):
        shutil.rmtree(folder, ignore_errors=True)
    (ROOT / "tp-live-in" / args.session).mkdir(parents=True)


def post(path, body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(args.base + path, data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=900)


with post("/tokenize", {"model": args.model, "messages": json.loads(args.messages.read_text()), "add_generation_prompt": True, "return_token_strs": True,
                        "chat_template_kwargs": {"enable_thinking": False}}) as resp:
    tokenized = json.loads(resp.read())
ids, strs, span_start = tokenized["tokens"], tokenized.get("token_strs"), 0
prompt = [args.placeholder_id] * args.reserve + ids
if strs and any("DRIFT" in piece for piece in strs):               # the system prompt frames the span: put the placeholders where @@DRIFT@@ stands
    text, spans, at = "", [], 0
    for piece in strs:
        spans.append((len(text), len(text) + len(piece))); text += piece
    a = text.index("@@DRIFT@@"); b = a + len("@@DRIFT@@")
    hit = [i for i, (x, y) in enumerate(spans) if x < b and y > a]
    span_start = hit[0]
    prompt = ids[:hit[0]] + [args.placeholder_id] * args.reserve + ids[hit[-1] + 1:]
if args.max_prompt_tokens is not None and len(prompt) > args.max_prompt_tokens:
    raise ValueError("prompt token limit exceeded before submission")
body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_new, "temperature": 0, "stream": True,
        "cache_salt": f"drift:{uuid.uuid4().hex}", "vllm_xargs": {"skip_writing_prefix_cache": 1}}
if not args.no_link:
    body["kv_transfer_params"] = {"drift_session": args.session, "drift_reserve": args.reserve, "drift_reserve_start": span_start}
    if args.no_tap:
        body["kv_transfer_params"]["drift_tap"] = False
if args.wait_publication:
    print(json.dumps({"publication_ready": True}), flush=True)
    wait_release(sys.stdin.fileno(), args.ready_timeout)
started = time.time()
print(json.dumps({"t": 0.0, "started": True, "own_prompt_tokens": len(ids), "span_start": span_start}), flush=True)
completed = False
with post("/v1/completions", body) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if line == "data: [DONE]":
            completed = True
            break
        if line.startswith("data:"):
            print(json.dumps({"t": round(time.time() - started, 2), "text": json.loads(line[5:])["choices"][0]["text"]}), flush=True)
if not completed:
    raise RuntimeError("GLM completion stream ended before its completion marker")
print(json.dumps({"t": round(time.time() - started, 2), "done": True}), flush=True)
