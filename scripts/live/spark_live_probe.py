"""Spark host: one LIVE session on the running GLM server (DriftGlm53Connector). The request streams; after
--delay-tokens generated tokens a memory file is released into the session's inbox, and the connector writes it into
the reserved span while GLM keeps decoding. Reports the text before/after and the decode-time taps that appeared.
Stdlib + numpy. The API key comes from the environment and is never printed."""
import argparse, json, os, shutil, time, urllib.request, uuid
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--question", required=True)
parser.add_argument("--system", default="")
parser.add_argument("--memory", type=Path, help="npz of l<layer> [rows,512]; released after --delay-tokens (must already be staged on the other ranks under the same session name)")
parser.add_argument("--session", required=True)
parser.add_argument("--reserve", type=int, default=1024)
parser.add_argument("--delay-tokens", type=int, default=0)
parser.add_argument("--max-new", type=int, default=200)
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--base", default="http://127.0.0.1:8888")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
inbox, outbox = ROOT / "tp-live-in" / args.session, ROOT / "tp-live-out" / args.session
shutil.rmtree(inbox, ignore_errors=True); shutil.rmtree(outbox, ignore_errors=True); inbox.mkdir(parents=True)


def post(path, body, stream=False):
    req = urllib.request.Request(args.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return urllib.request.urlopen(req, timeout=900)


messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": args.question}]
with post("/tokenize", {"model": args.model, "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}) as resp:
    ids = json.loads(resp.read())["tokens"]
body = {"model": args.model, "prompt": [args.placeholder_id] * args.reserve + ids, "max_tokens": args.max_new, "temperature": 0, "stream": True,
        "cache_salt": f"drift:{uuid.uuid4().hex}", "vllm_xargs": {"skip_writing_prefix_cache": 1}, "kv_transfer_params": {"drift_session": args.session, "drift_reserve": args.reserve}}
started, pieces, released_at, before = time.time(), [], None, ""


def release():
    tmp = inbox / ".000000.tmp.npz"
    shutil.copyfile(args.memory, tmp); os.replace(tmp, inbox / "000000.npz")


if args.memory and args.delay_tokens == 0:
    release(); released_at = 0.0
with post("/v1/completions", body) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        pieces.append(json.loads(line[5:])["choices"][0]["text"])
        if args.memory and released_at is None and len(pieces) >= args.delay_tokens:
            before = "".join(pieces); release(); released_at = round(time.time() - started, 2)
time.sleep(0.3)
taps = sorted(outbox.glob("[0-9]*.npz"))
spans = [(int(z["start"]), int(z["stop"])) for z in (np.load(t) for t in taps)]
first = np.load(taps[0]) if taps else None
print(json.dumps({"own_prompt_tokens": len(ids), "reserve": args.reserve, "seconds": round(time.time() - started, 2), "memory_released_at_s": released_at,
                  "text_before_memory": before, "text_after_memory": "".join(pieces)[len(before):], "chunks": len(pieces),
                  "taps": len(taps), "tap_spans": spans[:3] + (["..."] if len(spans) > 4 else []) + spans[-1:], "tap_rms_l3": None if first is None else round(float(np.sqrt((first["l3"].astype(np.float32) ** 2).mean())), 3),
                  "finished_marker": (outbox / "finished").read_text() if (outbox / "finished").exists() else None,
                  "errors": {p.name: p.read_text()[:300] for p in outbox.glob("error.*")}}, indent=1))
