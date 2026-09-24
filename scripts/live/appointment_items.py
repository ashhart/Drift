"""Write appointment scenarios as items for the offline halves of the live loop.

Each item carries one scenario's chats from appointment_scenarios.messages: GLM's linked messages, Qwen's messages, and
the true street and shop. studio_replay_memory.py turns Qwen's prompt into the memory the loop would publish, and
glm_export_live_taps.py --generate lets GLM write its reply against that memory while the connector taps, as the live
session does. Ids are s{seed}x{n}, so items from several seeds can share one folder.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.live.appointment_scenarios import VOCABULARY, messages, scenarios

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--count", type=int, default=12, help="scenarios per seed")
parser.add_argument("--vocabulary", choices=sorted(VOCABULARY), default="fresh")
parser.add_argument("--balance", action="store_true")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.vocabulary == "confirm2":
    parser.error("confirm2 names are reserved for the causal confirmation")
items = []
for seed in args.seeds:
    for n, scenario in enumerate(scenarios(seed, args.count, args.balance, args.vocabulary)):
        chats = messages(scenario)
        items.append({"id": f"s{seed}x{n}", "street": scenario["street"], "shop": scenario["shop"], "glm": chats["glm"], "qwen": chats["qwen"]})
args.out.write_text(json.dumps(items, indent=1) + "\n")
print(json.dumps({"items": len(items), "vocabulary": args.vocabulary}))
