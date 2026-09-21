"""Offline, training-only extraction for ONE explicit directional layer pair."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.manifest import inventory, sha256_file
from drift.train.alignment import shared_causal_endpoints

parser = argparse.ArgumentParser()
parser.add_argument("--source-model", type=Path, required=True)
parser.add_argument("--target-model", type=Path, required=True)
parser.add_argument("--source-layer", type=int, required=True)
parser.add_argument("--target-layer", type=int, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--max-rows", type=int, default=8192)
args = parser.parse_args()
if args.max_tokens < 2 or args.max_rows < 2:
    parser.error("positive nontrivial token/row budgets required")
if args.out.exists() or args.out.with_suffix(".json").exists():
    parser.error("output exists")
from transformers import AutoTokenizer
source = FrozenDecoder.from_local_hf(args.source_model, args.device)
target = FrozenDecoder.from_local_hf(args.target_model, args.device)
if not 0 <= args.source_layer < len(source.layers) or not 0 <= args.target_layer < len(target.layers):
    parser.error("layer outside checkpoint")
tokenizers = [AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                            trust_remote_code=False, use_fast=True)
              for path in (args.source_model, args.target_model)]
if not all(t.is_fast for t in tokenizers):
    parser.error("validated original-text offsets require a fast tokenizer")
collected = {name: [] for name in ("source_k", "source_v", "target_k", "target_v")}
seen, accepted, count, skipped = 0, 0, 0, []
with args.input.open() as handle, torch.no_grad():
    for line in handle:
        record = json.loads(line)
        if set(record) != {"id", "text", "split"} or record["split"] != "train":
            parser.error("training JSONL requires exactly id/text/split with split=train")
        seen += 1
        text = record["text"]
        batches = [t(text, add_special_tokens=False, return_offsets_mapping=True) for t in tokenizers]
        lengths = [len(x["input_ids"]) for x in batches]
        if min(lengths) == 0 or max(lengths) > args.max_tokens:
            skipped.append({"id": record["id"], "reason": "token_budget_or_empty"}); continue
        for tokenizer, batch in zip(tokenizers, batches):
            decoded = tokenizer.decode(batch["input_ids"], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if decoded != text:
                parser.error("tokenizer normalization/round-trip differs: implement and audit a text-coordinate adapter")
        aligned = shared_causal_endpoints(text, batches[0]["offset_mapping"], batches[1]["offset_mapping"])
        aligned = aligned[:args.max_rows - count]
        if not aligned:
            skipped.append({"id": record["id"], "reason": "no_shared_endpoint"}); continue
        outs = [worker.forward(torch.tensor(batch["input_ids"], dtype=torch.long))
                for worker, batch in zip((source, target), batches)]
        for side, layer, column, output in (("source", args.source_layer, 0, outs[0]),
                                            ("target", args.target_layer, 1, outs[1])):
            rows = torch.tensor([pair[column] for pair in aligned], device=output.logits.device)
            kv = output.canonical_delta[layer]
            for field in ("k", "v"):
                collected[f"{side}_{field}"].append(getattr(kv, field)[rows].detach().cpu().numpy())
        count += len(aligned); accepted += 1
        if count >= args.max_rows:
            break
if count < 2:
    parser.error("insufficient causally aligned rows")
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("wb") as handle:
    np.savez(handle, **{name: np.concatenate(values) for name, values in collected.items()})
manifest = {"mode": "offline_kv_regression", "text_mode": "raw_no_chat_template",
            "split": "train", "input_sha256": sha256_file(args.input),
            "array_sha256": sha256_file(args.out), "source_layer": args.source_layer,
            "target_layer": args.target_layer, "rows": count, "records_seen": seen,
            "records_accepted": accepted, "skipped": skipped,
            "source": inventory(args.source_model), "target": inventory(args.target_model)}
args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(args.out)
