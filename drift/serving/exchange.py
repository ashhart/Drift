"""Canonical-entry exchange files between a serving connector and the translator sidecar.

The server process only taps and injects; translators, pool banks and wire v2 live in a
sidecar. Both sides exchange canonical entries as safetensors plus a small JSON manifest.
Writes are atomic (temp + rename). No text, token ids or labels are stored: only tensors,
layer indices, positions and typed counters.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
import torch
from safetensors.torch import load_file, save_file
from drift.core.types import KV


def save_entries(directory: Path, name: str, entries: Mapping[int, KV | torch.Tensor], positions: torch.Tensor,
                 meta: Mapping[str, int | str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {"positions": positions.detach().cpu().to(torch.int64).contiguous()}
    layout = {}
    for layer, entry in entries.items():
        if isinstance(entry, KV):
            entry.check()
            tensors[f"layer{layer}.k"] = entry.k.detach().cpu().contiguous()
            tensors[f"layer{layer}.v"] = entry.v.detach().cpu().contiguous()
            layout[str(layer)] = "kv_split"
        else:
            if entry.ndim != 2 or not torch.isfinite(entry).all():
                raise ValueError("latent entries must be finite [T, C]")
            tensors[f"layer{layer}.latent"] = entry.detach().cpu().contiguous()
            layout[str(layer)] = "mla_latent"
    for key, value in meta.items():
        if not isinstance(value, (int, str)) or (isinstance(value, str) and len(value) > 128):
            raise ValueError("exchange metadata carries typed counters and short identifiers only")
    target, temp = directory / f"{name}.safetensors", directory / f".{name}.{os.getpid()}.tmp"
    save_file(tensors, str(temp))
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    os.replace(temp, target)
    manifest_temp = directory / f".{name}.{os.getpid()}.json.tmp"
    manifest_temp.write_text(json.dumps({"layout": layout, "tokens": int(positions.numel()), "sha256": digest, **dict(meta)}, sort_keys=True))
    os.replace(manifest_temp, directory / f"{name}.json")     # the manifest appears last: readers key on it
    return target


def load_entries(directory: Path, name: str, device: str | torch.device = "cpu") -> tuple[dict[int, KV | torch.Tensor], torch.Tensor, dict]:
    manifest = json.loads((directory / f"{name}.json").read_text())
    path = directory / f"{name}.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("exchange file hash mismatch")
    tensors = load_file(str(path), device=str(device))
    entries: dict[int, KV | torch.Tensor] = {}
    for layer, kind in manifest["layout"].items():
        if kind == "kv_split":
            entries[int(layer)] = KV(tensors[f"layer{layer}.k"], tensors[f"layer{layer}.v"])
        else:
            entries[int(layer)] = tensors[f"layer{layer}.latent"]
    positions = tensors["positions"]
    for entry in entries.values():
        n = entry.tokens if isinstance(entry, KV) else entry.shape[0]
        if n != manifest["tokens"] or n != positions.numel():
            raise ValueError("exchange file is inconsistent across layers")
    return entries, positions, manifest


def ready(directory: Path, name: str) -> bool:
    return (directory / f"{name}.json").exists()
