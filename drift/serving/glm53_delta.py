"""Delta exports from GLM-5.3's handoff connector, and joining one to the full export of the prompt's first part.

The connector computes a full export's whole prompt in one engine step, which stalls on the Sparks at 121k tokens. A
long prompt is read in two requests instead: a full export of its first part, then the whole prompt with export_from N,
which resumes from the prefix cache and exports only the blocks from N rounded down to the cache block
(scripts/live/spark_dropin_export.py --split-at). glm53_handoff.read_latents refuses delta exports; this module reads
them and stitches the two. NumPy only.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from drift.serving.glm53_handoff import _ATTN, dequantize_fp8_ds_mla, read_header, read_header_bytes


def read_latents_span(path, skip_draft_layers: bool = True) -> tuple[int, dict[int, np.ndarray]]:
    """A full or delta export's latents: (first position, per MLA layer float32 [n_tokens - first, 512])."""
    in_memory = not isinstance(path, (str, Path))
    header, data_start = read_header_bytes(path) if in_memory else read_header(path)
    if header.get("format") != "glm53-handoff-raw-v1":
        raise ValueError(f"unsupported blob format {header.get('format')}")
    if header.get("cache_config", {}).get("cache_dtype") != "fp8_ds_mla":
        raise ValueError("this reader is qualified for fp8_ds_mla caches only")
    n_tokens = int(header["n_tokens"])
    raw = np.frombuffer(path, dtype=np.uint8)[data_start:] if in_memory else np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)
    out: dict[int, np.ndarray] = {}
    starts = set()
    for tensor in header["tensors"]:
        name = tensor.get("layer", "")
        match = _ATTN.search(name)
        if not match or "skipped" in tensor or tensor.get("kind") in {"state", "capture"}:
            continue
        if skip_draft_layers and not name.startswith("language_model."):
            continue                                          # drafter layers (model.layers.45+) are not target state
        pages, slots, width = tensor["shape"]
        if tensor["dtype"] != "uint8" or width != 656:
            raise ValueError(f"layer {name}: expected fp8_ds_mla pages, got {tensor['dtype']} {tensor['shape']}")
        first = int(tensor.get("first_block_index", 0)) * int(tensor.get("pages_per_block", 1)) * slots
        if first >= n_tokens or pages * slots < n_tokens - first:
            raise ValueError("exported pages do not cover the prompt")
        block = np.frombuffer(raw[tensor["offset"]: tensor["offset"] + tensor["nbytes"]], dtype=np.uint8).reshape(pages * slots, width)
        out[int(match.group(1))] = dequantize_fp8_ds_mla(np.ascontiguousarray(block[:n_tokens - first]))
        starts.add(first)
    if not out:
        raise ValueError("no MLA attention layers found in the blob")
    if len(starts) != 1:
        raise ValueError("layers start at different positions")
    return starts.pop(), dict(sorted(out.items()))


def stitch(prefix: dict[int, np.ndarray], first: int, delta: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """A full export of a prompt's first tokens joined to a later delta export of the whole prompt at `first`."""
    if sorted(prefix) != sorted(delta) or any(len(v) < first for v in prefix.values()):
        raise ValueError("the prefix export must hold the same layers and reach the delta's first position")
    return {layer: np.concatenate((prefix[layer][:first], delta[layer])) for layer in sorted(prefix)}
