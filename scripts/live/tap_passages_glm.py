"""Tap GLM (live server) on every passage of a jsonl file; resumable. Output: <out>/taps_glm/<passage id>.npz"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import glm_read, passage_id
parser = argparse.ArgumentParser()
parser.add_argument("--texts", type=Path, nargs="+", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--chunk", type=int, default=300)
args = parser.parse_args()
texts = {}
for path in args.texts:
    for line in path.read_text().splitlines():
        t = json.loads(line)["text"]; texts[passage_id(t)] = t
args.out.mkdir(parents=True, exist_ok=True)
todo = {k: v for k, v in texts.items() if not (args.out / "taps_glm" / f"{k}.npz").exists()}
keys = list(todo)
for start in range(0, len(keys), args.chunk):
    glm_read({k: todo[k] for k in keys[start:start + args.chunk]}, args.out, f"{args.out.name}-{start}")
print(json.dumps({"passages": len(texts), "tapped_now": len(todo)}))
