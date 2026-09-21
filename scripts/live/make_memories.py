"""Build reader memories for a passage file under several forward-translator variants (for qa_ab.py)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import GLM_LAYERS, glm_read, load_reader, passage_id
from drift.translate.fanout import FanoutReader

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--base", type=Path, default=Path("local/live/stacked2.npz"))
parser.add_argument("--fanout", type=Path)
parser.add_argument("--gain-power", type=float, default=1.5)
args = parser.parse_args()
passages = {passage_id(json.loads(l)["text"]): json.loads(l)["text"] for l in args.passages.read_text().splitlines()}
args.out.mkdir(parents=True, exist_ok=True)
taps = args.out / "taps_glm"
if not taps.exists() or len(list(taps.glob("*.npz"))) < len(passages):
    glm_read(passages, args.out, args.out.name)
base = load_reader(args.base)
variants = {"base": base, **({"fanout": FanoutReader.load(args.fanout, base)} if args.fanout else {})}
extra = 0
for name, reader in variants.items():
    (args.out / name).mkdir(exist_ok=True)
    for pid in passages:
        z = np.load(taps / f"{pid}.npz")
        entries = reader.read({l: z[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, args.gain_power)
        if name == "fanout":
            extra += next(iter(entries.values()))[0].shape[0] - z["l3"].shape[0]
        np.savez(args.out / name / f"{pid}.npz", **{f"k{l}": k.astype(np.float16) for l, (k, v) in entries.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in entries.items()})
print(json.dumps({"passages": len(passages), "variants": list(variants), "extra_entries_from_fanout": extra}))
