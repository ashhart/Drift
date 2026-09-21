"""Level-2 adapter parity on a local checkpoint at its deployed precision (docs/ADAPTERS.md).

The tolerance must be preregistered in a JSON file BEFORE this runs; the script refuses
to run without it and never widens it. It reports the full error distribution, not only
a pass flag, and writes registry evidence (qualification/level2.json) plus hashes.

Usage:
  PYTHONPATH=. .venv-next/bin/python scripts/parity_real.py --family qwen4_exp --checkpoint DIR \
      --prereg configs/preregistration.json --registry registry --model-id ID --device cpu \
      --lengths 1,2,7,16,127,256 --seeds 0,1,2
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=["qwen4_exp", "glm5_next"], required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--registry", type=Path, required=True)
parser.add_argument("--model-id", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--lengths", default="1,2,7,16,127,256")
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--continuation", type=int, default=5)
args = parser.parse_args()

prereg = json.loads(args.prereg.read_text())
tolerance = prereg.get("parity", {}).get("level2_real_weight_tolerance")
if not isinstance(tolerance, (int, float)):
    print("BLOCKED: preregister a numeric parity.level2_real_weight_tolerance before running", file=sys.stderr)
    raise SystemExit(2)
prereg_sha = hashlib.sha256(args.prereg.read_bytes()).hexdigest()
dtype = getattr(torch, args.dtype)

if args.family == "qwen4_exp":
    from drift.adapters.qwen4_exp import Qwen4ExpAdapter as Adapter
else:
    from drift.adapters.glm5_next import Glm5NextAdapter as Adapter
started = time.perf_counter()
adapter = Adapter.from_local(str(args.checkpoint), args.device, dtype)
model = adapter.model
vocab = int(getattr(model.config, "vocab_size", None) or model.config.get_text_config().vocab_size)
digest_before = adapter.frozen_digest()


def stock_forward(ids):
    out = model(input_ids=ids[None].to(adapter.device), use_cache=False)
    return (out.logits if hasattr(out, "logits") else out.last_hidden_state)[0].float()


records, worst = [], 0.0
with torch.no_grad():
    for seed in (int(s) for s in args.seeds.split(",")):
        g = torch.Generator().manual_seed(seed)
        for n in (int(x) for x in args.lengths.split(",")):
            ids = torch.randint(3, vocab, (n + args.continuation,), generator=g)
            reference = stock_forward(ids)
            full = adapter.forward(ids).output.float()
            prefix = adapter.forward(ids[:n])
            continued = adapter.forward(ids[n:], prefix.state).output.float()
            closed = adapter.forward(ids[n:], prefix.state, foreign=None, override=0.0).output.float()
            errs = {
                "full_vs_stock": (full - reference).abs(),
                "continuation_vs_stock": (continued - reference[n:]).abs(),
            }
            row = {"seed": seed, "length": n, "continuation": args.continuation,
                   "hard_off_bit_identical": bool(torch.equal(closed, continued))}
            for name, e in errs.items():
                row[name] = {"max": float(e.max()), "mean": float(e.mean()),
                             "p50": float(e.median()), "p99": float(e.flatten().kthvalue(max(1, int(0.99 * e.numel()))).values)}
                worst = max(worst, row[name]["max"])
            records.append(row)
passed = worst <= tolerance and all(r["hard_off_bit_identical"] for r in records) and adapter.frozen_digest() == digest_before
report = {
    "level": 2, "family": args.family, "checkpoint": str(args.checkpoint), "dtype": args.dtype, "device": args.device,
    "preregistered_tolerance": tolerance, "preregistration_sha256": prereg_sha,
    "config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
    "weights_manifest_sha256": hashlib.sha256("".join(sorted(
        hashlib.sha256(p.read_bytes()).hexdigest() for p in args.checkpoint.glob("*.safetensors"))).encode()).hexdigest(),
    "worst_max_abs": worst, "records": records, "weights_unchanged": adapter.frozen_digest() == digest_before,
    "wall_seconds": time.perf_counter() - started, "status": "PASSED" if passed else "FAILED",
}
out_dir = args.registry / args.model_id / "qualification"
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / "level2.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
print("evidence:", path, hashlib.sha256(path.read_bytes()).hexdigest())
raise SystemExit(0 if passed else 1)
