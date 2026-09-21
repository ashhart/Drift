"""Spark host: how does the running server split a prefill into engine steps? (read-only probe through the
owner's streaming handoff export). Prints cache geometry and the step boundaries for a prompt of --tokens."""
import argparse, json, os, shutil, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_handoff import read_header

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=1100)
parser.add_argument("--token-id", type=int, default=198)
args = parser.parse_args()
root, hid = Path("/dev/shm/glm53-handoff"), f"drift-chunkprobe-{int(time.time())}"
body = json.dumps({"model": "GLM-5.3-Flash-EXL3", "prompt": [args.token_id] * args.tokens, "max_tokens": 1, "temperature": 0,
                   "kv_transfer_params": {"glm53_handoff": True, "handoff_id": hid, "stream": True}}).encode()
req = urllib.request.Request("http://127.0.0.1:8888/v1/completions", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['DRIFT_GLM_KEY']}"})
with urllib.request.urlopen(req, timeout=300) as resp:
    usage = json.loads(resp.read()).get("usage")
for _ in range(400):
    if (root / hid / "rank0.ready").exists():
        break
    time.sleep(0.05)
header, _ = read_header(root / hid / "rank0.bin")
groups = [{"group": g["group"], "layers": len(g["layers"]), "first": g["layers"][0], "spec": {k: v for k, v in g["spec"].items() if k in ("type", "block_size", "compress_ratio", "sliding_window", "page_size_bytes", "num_speculative_blocks")}} for g in header["groups"]]
print(json.dumps({"usage": usage, "n_tokens": header["n_tokens"], "cache_config": header["cache_config"], "boundary_tokens": header["boundary_tokens"], "states": header["states"],
                  "segments": [{k: s.get(k) for k in ("computed", "upto", "tokens", "blocks") if k in s} for s in header["segments"]][:12], "files": sorted(p.name for p in (root / hid).iterdir()),
                  "block_ids_per_group": [len(b) for b in header["block_ids"]], "groups": groups}, indent=1))
shutil.rmtree(root / hid, ignore_errors=True)
