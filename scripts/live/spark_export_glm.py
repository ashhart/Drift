"""GLM (live vLLM server) reads each text and its connector exports the cache; the export is LEFT IN PLACE for an RDMA pull by
the Studio's handoffd (Drift over MCDMA). Runs on the Spark head. Prints one JSON line per record; the key is never printed."""
import argparse, json, os, shutil, sys, time, urllib.request
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--ids", type=Path, required=True, help="json {records: [{id, ids_glm}]}")
parser.add_argument("--handoff-root", type=Path, default=Path("/dev/shm/glm53-handoff"))
parser.add_argument("--url", default="http://127.0.0.1:8888/v1/completions")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
key = os.environ["DRIFT_GLM_KEY"]
for r in json.loads(args.ids.read_text())["records"]:
    hid = f"drift-{r['id']}"
    shutil.rmtree(args.handoff_root / hid, ignore_errors=True)
    body = json.dumps({"model": args.model, "prompt": r["ids_glm"], "max_tokens": 1, "temperature": 0, "kv_transfer_params": {"glm53_handoff": True, "handoff_id": hid}}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=300) as resp:
        json.loads(resp.read())
    t_request = time.time() - t0
    ready = args.handoff_root / hid / "rank0.ready"
    while not ready.exists() and time.time() - t0 < 120:
        time.sleep(0.01)
    if not ready.exists():
        raise TimeoutError(f"no export for {hid}")
    blob = args.handoff_root / hid / "rank0.bin"
    print(json.dumps({"id": r["id"], "remote": str(blob), "bytes": blob.stat().st_size, "tokens": len(r["ids_glm"]), "request_s": round(t_request, 3), "export_ready_s": round(time.time() - t0, 3)}), flush=True)
