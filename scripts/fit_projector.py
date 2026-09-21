"""Fit ONE directional layer pair from already aligned training-only arrays."""
import argparse
from pathlib import Path
import numpy as np
import torch
from drift.core.types import KV
from drift.core.projector import BridgeProjector
from drift.train.fit import fit_mlp, save_projector
from drift.runtime.manifest import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--kind", choices=["ridge", "mlp"], default="ridge")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()
with np.load(args.data, allow_pickle=False) as file:
    if set(file.files) != {"source_k", "source_v", "target_k", "target_v"}:
        parser.error("expected exactly source_k/source_v/target_k/target_v")
    tensors = {name: torch.from_numpy(file[name].astype(np.float32)) for name in file.files}
source, target = KV(tensors["source_k"], tensors["source_v"]), KV(tensors["target_k"], tensors["target_v"])
source.check(); target.check()
torch.manual_seed(args.seed)
projector = BridgeProjector(tuple(source.k.shape[1:]), tuple(target.k.shape[1:]), args.kind)
if args.kind == "ridge":
    projector.fit(source, target)
    report = {"kind": "ridge_kv_regression", "training_rows": source.tokens}
else:
    report = fit_mlp(projector, source, target, steps=args.steps, seed=args.seed)
report.update({"data_sha256": sha256_file(args.data), "seed": args.seed,
               "behavior_distillation": "NOT_RUN", "semantic_transfer": "NOT_EVALUATED"})
save_projector(projector, args.out, report)
print(args.out)
