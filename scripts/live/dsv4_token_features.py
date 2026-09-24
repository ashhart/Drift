"""Turn DeepSeek V4 page captures into per-token features, the sender side of translators from DeepSeek's cache.

Reads the captures of dsv4_capture_rows.py, one npz per passage, and decodes every ratio-4 layer's compressed entries
with drift.translate.dsv4_pages. DeepSeek keeps one ratio-4 entry per group of four tokens, so each passage token gets
the 448 dequantized values of its group's entry from every such layer, in layer order; the rope values carry position,
not content, and are left out. --indexer adds each layer's 128-value indexer key for the group, and --context N adds the
entries of the N groups before and after (zeros past the tapped span). Writes one npz per passage with x [tokens,
width] as float16 and the tokens' character offsets, the layout fit_state_translator.py reads with --source-key x.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
from drift.translate.dsv4_pages import FP8_DIMS, entries, index_keys

parser = argparse.ArgumentParser()
parser.add_argument("--captures", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--indexer", action="store_true", help="add each ratio-4 layer's indexer key for the token's group")
parser.add_argument("--context", type=int, default=0, help="add the entries of this many neighbouring groups on each side")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
LAYER = re.compile(r"model\.layers\.(\d+)\.attn(\.indexer\.k_cache)?$")
written, layers, width = 0, None, 0
for path in sorted(args.captures.glob("*.npz")):
    z = np.load(path)
    manifest = json.loads(bytes(z["manifest"]).decode())
    start, tapped, count = (int(v) for v in z["span"])
    groups, keys = {}, {}
    for i, entry in enumerate(manifest):
        match = LAYER.match(entry["name"])
        if entry["ratio"] == 4 and match:
            (keys if match.group(2) else groups)[int(match.group(1))] = i
    if layers is None:
        layers = sorted(groups)
    elif sorted(groups) != layers or (args.indexer and sorted(keys) != layers):
        raise SystemExit(f"{path.name}: its ratio-4 layers differ from the first capture's")
    per_group = [entries(z[f"r{groups[layer]}"], 256 // 4)[:, :FP8_DIMS] for layer in layers]
    if any(len(r) != tapped // 4 for r in per_group):
        raise SystemExit(f"{path.name}: entries do not cover the tapped span")
    table = np.concatenate(per_group, axis=1)                          # [groups, layers * 448]
    parts = [table]
    for shift in range(1, args.context + 1):                            # neighbours, zero past the span's ends
        before, after = np.zeros_like(table), np.zeros_like(table)
        before[shift:], after[:-shift] = table[:-shift], table[shift:]
        parts += [before, after]
    if args.indexer:
        parts.append(np.concatenate([index_keys(z[f"r{keys[layer]}"], 256 // 4) for layer in layers], axis=1))
    table = np.concatenate(parts, axis=1)
    width = table.shape[1]
    x = table[np.arange(count) // 4].astype(np.float16)                # the span starts on a page, so groups start at its first token
    np.savez(args.out / path.name, x=x, offsets=z["offsets"])
    written += 1
print(json.dumps({"written": written, "layers": layers, "width": width}))
