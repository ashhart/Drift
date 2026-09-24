"""Qwen3.8-Flash-Next cache pages on vLLM: write Drift memory into a request's reserved span.

Layouts, from the image's vLLM source (models/qwen3_8_flash_next/nvidia/qsa.py and indexer_qsa.py):
  main KV     "...layers.{L}.self_attn.attn" [blocks, kv_heads_per_rank, slots, 2 * head_dim] bf16;
              a token's row is K (after k_norm and RoPE) then V, and KV head h of the model lives on rank h.
  compressed  "...layers.{L}...compressed_key_cache" [blocks, groups_per_block, 1, index_dim] bf16; group g holds
              RoPE(first position) of the k_layernorm of the mean raw key over tokens [ratio * g, ratio * (g + 1)).
The MTP layer is not Drift memory and is never written.
"""
from __future__ import annotations
import re
import numpy as np

HEAD_DIM, INDEX_DIM = 256, 128
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_MTP = re.compile(r"(?:^|\.)mtp\.")


def _main_layers(names, suffix: str) -> dict[int, str]:
    found = {}
    for name in names:
        match = _LAYER.search(name)
        if match and name.endswith(suffix) and not _MTP.search(name):
            layer = int(match.group(1))
            if layer in found:
                raise ValueError(f"layer {layer} has two {suffix} caches")
            found[layer] = name
    return found


def attention_layers(names) -> dict[int, str]:
    """Main-model full-attention caches by layer index."""
    return _main_layers(names, "self_attn.attn")


def compressed_layers(names) -> dict[int, str]:
    """Main-model compressed selector-key caches by layer index."""
    return _main_layers(names, "compressed_key_cache")


def rotate(keys: np.ndarray, positions: np.ndarray, theta: float, rotary_dim: int) -> np.ndarray:
    """Split-half RoPE on the first rotary_dim dims of keys [tokens, heads, dim], float32 as in omlx_cache.Rope."""
    if keys.ndim != 3 or len(positions) != keys.shape[0] or not 0 < rotary_dim <= keys.shape[2] or rotary_dim % 2:
        raise ValueError("keys must be [tokens, heads, dim] with one position per token and an even rotary_dim")
    inv = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    phase = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    phase = np.concatenate((phase, phase), axis=-1).astype(np.float32)
    cos, sin = np.cos(phase)[:, None, :], np.sin(phase)[:, None, :]
    x = keys.astype(np.float32)
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    turned = np.concatenate((-rot[..., half:], rot[..., :half]), axis=-1)
    return np.concatenate((rot * cos + turned * sin, rest), axis=-1)


def slots(page_rows, per_page: int, first: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Page row and slot of each position in [first, first + count), from the request's page rows in order."""
    positions = np.arange(first, first + count)
    if first < 0 or count <= 0 or (positions[-1] // per_page) >= len(page_rows):
        raise ValueError("the request's pages do not cover the reserved span")
    return np.asarray(page_rows, dtype=np.int64)[positions // per_page], positions % per_page


def write_kv(part, page_rows, first: int, keys, values) -> None:
    """Write this rank's rotated keys and values [tokens, heads_per_rank, head_dim] into the span, in place."""
    import torch
    if part.dtype != torch.bfloat16 or part.ndim != 4 or part.shape[3] != 2 * HEAD_DIM:
        raise ValueError(f"main KV cache {tuple(part.shape)} {part.dtype} is not bf16 [blocks, heads, slots, {2 * HEAD_DIM}]")
    heads = int(part.shape[1])
    if keys.shape != values.shape or keys.ndim != 3 or keys.shape[1:] != (heads, HEAD_DIM):
        raise ValueError(f"memory rows {keys.shape} do not match {heads} heads of {HEAD_DIM}")
    rows, where = slots(page_rows, int(part.shape[2]), first, keys.shape[0])
    data = torch.from_numpy(np.concatenate((keys, values), axis=-1).astype(np.float32)).to(part.device, torch.bfloat16)
    index = torch.as_tensor(rows, device=part.device), torch.as_tensor(where, device=part.device)
    for head in range(heads):
        part[index[0], head, index[1]] = data[:, head]


def write_compressed(part, page_rows, first_group: int, keys) -> None:
    """Write compressed selector keys [groups, index_dim] for groups [first_group, first_group + groups), in place."""
    import torch
    if part.dtype != torch.bfloat16 or part.ndim != 4 or part.shape[2:] != (1, INDEX_DIM):
        raise ValueError(f"compressed key cache {tuple(part.shape)} {part.dtype} is not bf16 [blocks, groups, 1, {INDEX_DIM}]")
    if keys.ndim != 2 or keys.shape[1] != INDEX_DIM:
        raise ValueError(f"compressed keys {keys.shape} are not [groups, {INDEX_DIM}]")
    rows, where = slots(page_rows, int(part.shape[1]), first_group, keys.shape[0])
    data = torch.from_numpy(np.asarray(keys, dtype=np.float32)).to(part.device, torch.bfloat16)
    part[torch.as_tensor(rows, device=part.device), torch.as_tensor(where, device=part.device), 0] = data
