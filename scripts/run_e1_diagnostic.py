"""Small offline E1 DIAGNOSTIC, one mapped layer; NOT a preregistered science run.

Input fields: id, context, question. No answer key. Outputs floor, reread ceiling,
foreign-memory, hard-closed gate, and wrong-context control. Raw text prompts are
explicitly a diagnostic formatting choice. OMP, mailbox, and MCDMA are not used.
"""
import argparse
import json
from pathlib import Path
from uuid import uuid4
import torch
from drift.core.types import Delta
from drift.core.memory import ForeignKVBank
from drift.runtime.decoder import FrozenDecoder
from drift.train.fit import load_projector
from drift.runtime.manifest import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument("--source-model", type=Path, required=True)
parser.add_argument("--target-model", type=Path, required=True)
parser.add_argument("--source-layer", type=int, required=True)
parser.add_argument("--target-layer", type=int, required=True)
parser.add_argument("--projector", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--new-tokens", type=int, default=32)
parser.add_argument("--foreign-window", type=int, default=128)
parser.add_argument("--source-max-tokens", type=int, default=512)
args = parser.parse_args()
if args.out.exists():
    parser.error("output exists")
from transformers import AutoTokenizer
source = FrozenDecoder.from_local_hf(args.source_model, args.device)
target = FrozenDecoder.from_local_hf(args.target_model, args.device)
projector = load_projector(args.projector, args.device)
if not 0 <= args.source_layer < len(source.layers) or not 0 <= args.target_layer < len(target.layers):
    parser.error("invalid layer pair")
if projector.source != (source.kvheads, source.dim) or projector.target != (target.kvheads, target.dim):
    parser.error("projector/checkpoint shape mismatch")
st, tt = [AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
          for path in (args.source_model, args.target_model)]
records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
if len(records) < 2 or any(set(row) != {"id", "context", "question"} for row in records):
    parser.error("at least two rows, exactly id/context/question; keep answer keys outside this runner")
if len({row["id"] for row in records}) != len(records):
    parser.error("duplicate example IDs")

def tokens(tokenizer, text):
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

@torch.no_grad()
def bank_for(text):
    ids = tokens(st, text)
    if not 1 <= ids.numel() <= args.source_max_tokens:
        raise ValueError("source context exceeds declared budget; no silent truncation")
    out = source.forward(ids)
    session = uuid4()
    bank = ForeignKVBank(session, 0, [(args.source_layer, args.target_layer, projector)],
                         sinks=4, recent=args.foreign_window)
    bank.commit(Delta(session, 0, 0, 0, 0, {args.source_layer: out.canonical_delta[args.source_layer]}))
    return bank.pin()

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("x") as output:
    for index, row in enumerate(records):
        question = "Question: " + row["question"] + "\nAnswer:"
        prompt = tokens(tt, question)
        bank, wrong = bank_for(row["context"]), bank_for(records[(index + 1) % len(records)]["context"])
        arms = {"floor": (prompt, None, 0.0),
                "reread_ceiling": (tokens(tt, row["context"] + "\n" + question), None, 0.0),
                "foreign": (prompt, bank, 1.0), "gate_closed": (prompt, bank, 0.0),
                "wrong_context": (prompt, wrong, 1.0)}
        result = {}
        for name, (ids, view, gate) in arms.items():
            generated = target.generate(ids, args.new_tokens, view, gate, tt.eos_token_id)
            result[name] = {"text": tt.decode(generated, skip_special_tokens=True),
                            "receiver_prompt_tokens": ids.numel(), "generated_tokens": len(generated)}
        if result["floor"]["text"] != result["gate_closed"]["text"]:
            raise AssertionError("hard-closed gate changed output")
        output.write(json.dumps({"id": row["id"], "mode": "E1_DIAGNOSTIC_NOT_PREREGISTERED",
                                 "arms": result}) + "\n")
meta = {"scientific_status": "NOT_ESTABLISHED", "backend": "inproc_direct_bank",
        "input_sha256": sha256_file(args.input), "output_sha256": sha256_file(args.out),
        "source_layer": args.source_layer, "target_layer": args.target_layer,
        "missing_for_formal_E1": ["complete model/run pin audit", "text-summary arm",
                                  "full cost ledger", "process isolation audit",
                                  "preregistered independent-unit scoring"]}
args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(args.out)
