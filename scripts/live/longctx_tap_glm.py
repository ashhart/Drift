"""GLM (live server) reads each long document once; its cache latents are exported and pulled. Resumable."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import glm_read
parser = argparse.ArgumentParser()
parser.add_argument("--docs", type=Path, required=True)
parser.add_argument("--only", nargs="*")
args = parser.parse_args()
out = args.docs.parent
docs = {d["id"]: d["text"] for d in json.loads(args.docs.read_text())["docs"] if not args.only or d["id"] in args.only}
todo = {k: v for k, v in docs.items() if not (out / "taps_glm" / f"{k}.npz").exists()}
t0 = time.time()
if todo:
    glm_read(todo, out, f"frontier-{out.name}-{int(t0)}")
print(json.dumps({"docs": len(docs), "tapped_now": len(todo), "seconds": round(time.time() - t0, 1)}))
