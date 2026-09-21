"""GLM (live server) taps for a corpus json (records with id + text), in resumable chunks. Output: <out>/<id>.npz"""
import argparse, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import glm_read
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--chunk", type=int, default=100)
args = parser.parse_args()
records = {r["id"]: r["text"] for r in json.loads(args.corpus.read_text())["records"]}
args.out.mkdir(parents=True, exist_ok=True)
todo = [k for k in records if not (args.out / f"{k}.npz").exists()]
t0 = time.time()
for start in range(0, len(todo), args.chunk):
    work = args.out.parent / f".glm_chunk_{start}"; work.mkdir(exist_ok=True)
    taps = glm_read({k: records[k] for k in todo[start:start + args.chunk]}, work, f"frontier-{args.out.name}-{int(t0)}-{start}")
    for f in taps.glob("*.npz"):
        shutil.move(str(f), args.out / f.name)
    shutil.rmtree(work)
    print("chunk", start, round(time.time() - t0), "s", flush=True)
print(json.dumps({"records": len(records), "tapped_now": len(todo), "seconds": round(time.time() - t0, 1)}))
