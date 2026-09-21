"""M0 identity correctness gate. Does not establish same-family/cross-family gains."""
import argparse
import json
from pathlib import Path
import torch
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.runtime.manifest import parameter_digest

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
parser.add_argument("--device", default="cpu")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists; do not overwrite evidence")
torch.set_num_threads(1)
worker = (FrozenDecoder(ToyModel()) if args.model is None
          else FrozenDecoder.from_local_hf(args.model, args.device))
ids = torch.tensor([1, 7, 3, 11, 2, 5, 13, 17], dtype=torch.long)
before = parameter_digest(worker.model)
with torch.no_grad():
    prefix = worker.forward(ids[:5])
    ordinary = worker.forward(ids[5:], prefix.state)
    imported = worker.import_self_prefix(prefix.canonical_delta, torch.arange(5))
    transplant = worker.forward(ids[5:], imported)
    full = worker.forward(ids)
    errors = {"self_handoff_max_abs": float((ordinary.logits - transplant.logits).abs().max()),
              "prefill_vs_incremental_max_abs": float((full.logits[5:] - ordinary.logits).abs().max())}
    if args.model is not None:
        stock = worker.model(input_ids=ids[None].to(worker.device), use_cache=False).logits[0]
        errors["stock_hf_forward_max_abs"] = float((stock - full.logits).abs().max())
unchanged = before == parameter_digest(worker.model)
passed = max(errors.values()) < 2e-5 and unchanged
report = {"fixture": "random_toy" if args.model is None else "local_checkpoint",
          "status": "PASSED" if passed else "FAILED", "errors": errors,
          "weights_unchanged": unchanged, "semantic_transfer": "NOT_EVALUATED"}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if passed else 1)
