"""Tap GLM-5.3 canonical latents from the LIVE vLLM server through the owner's handoff connector.
Runs on the Spark host (stdlib + numpy). API key comes from the environment and is never printed."""
import argparse, json, os, shutil, sys, time, urllib.request
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_handoff import read_latents

parser = argparse.ArgumentParser()
parser.add_argument("--ids", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--handoff-root", type=Path, default=Path("/dev/shm/glm53-handoff"))
parser.add_argument("--url", default="http://127.0.0.1:8888/v1/completions")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
key = os.environ["DRIFT_GLM_KEY"]
args.out.mkdir(parents=True, exist_ok=True)
records = json.loads(args.ids.read_text())["records"]
started, done, tokens = time.time(), 0, 0
for r in records:
    target = args.out / f"{r['id']}.npz"
    if target.exists():
        continue
    hid = f"drift-{r['id']}"
    shutil.rmtree(args.handoff_root / hid, ignore_errors=True)
    body = json.dumps({"model": args.model, "prompt": r["ids_glm"], "max_tokens": 1, "temperature": 0,
                       "kv_transfer_params": {"glm53_handoff": True, "handoff_id": hid}}).encode()
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        json.loads(resp.read())
    ready = args.handoff_root / hid / "rank0.ready"
    for _ in range(2400):                                        # long documents export several 3584-token blocks
        if ready.exists():
            break
        time.sleep(0.05)
    else:
        raise TimeoutError(f"no export for {hid}")
    latents = read_latents(args.handoff_root / hid / "rank0.bin")
    if next(iter(latents.values())).shape[0] != len(r["ids_glm"]):
        raise ValueError("export length does not match the prompt")
    np.savez(target, **{f"l{layer}": x.astype(np.float16) for layer, x in latents.items()})
    shutil.rmtree(args.handoff_root / hid, ignore_errors=True)
    done += 1; tokens += len(r["ids_glm"])
print(json.dumps({"tapped": done, "tokens": tokens, "seconds": round(time.time() - started, 1)}))
