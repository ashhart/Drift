"""Rebuild a live appointment run's forward follow-ups as offline gate items, with the GLM rows the loop actually received.

Run on the Studio after a demo_appointment.py run with --save-taps, the run folder copied next to the loop's out folder.
For each linked scenario it joins the saved forward taps in source order into one latent array per GLM layer, and writes
an item in the loop's follow-up layout: Qwen's conversation and its own answer, the framing line, GLM's block, the
follow-up question. studio_qwen_state_gate.py then answers the item from those exact latents, so its result differs from
the live answer only by what the loop itself did.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

FRAME = "My partner's message, from our shared memory:"                # must match studio_mcdma_loop.py
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True, help="the demo_appointment.py output folder")
parser.add_argument("--taps-root", type=Path, default=Path("out"), help="where the loop kept each session's taps")
parser.add_argument("--followup", required=True, help="the follow-up question the run asked")
parser.add_argument("--items", type=Path, required=True)
parser.add_argument("--latents", type=Path, required=True)
args = parser.parse_args()
args.latents.mkdir(parents=True, exist_ok=True)
items = []
for row in json.loads((args.run / "report.json").read_text())["rows"]:
    n = row["scenario"]
    report = json.loads((args.run / f"s{n}.linked" / "report.json").read_text())
    taps = sorted((args.taps_root / report["qwen"]["session"] / "taps").glob("*.npz"))
    if not taps:
        raise SystemExit(f"scenario {n}: no saved taps")
    parts, cursor = [], None
    for path in taps:
        tap = np.load(path)
        if cursor is not None and int(tap["start"]) != cursor:
            raise SystemExit(f"scenario {n}: saved taps are not contiguous")
        cursor = int(tap["stop"])
        parts.append({k: tap[k] for k in tap.files if k[0] == "l" and k[1:].isdigit()})
    np.savez(args.latents / f"r{n}.npz", **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    messages = json.loads((args.run / f"s{n}.qwen.json").read_text())
    conversation = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    items.append({"id": f"r{n}-replay", "passage": report["summary"]["glm"]["text"], "question": args.followup, "answer": row["shop"],
                  "live_answer": report["qwen"]["followup_answer"],
                  "head_text": f"{conversation}<|im_start|>assistant\n<think>\n\n</think>\n\n{report['qwen']['text']}<|im_end|>\n<|im_start|>user\n{FRAME}\n",
                  "tail_text": f"\n\n{args.followup}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"})
args.items.write_text(json.dumps(items, indent=1) + "\n")
print(json.dumps({"items": len(items)}))
