"""Reader for the owner's GLM-5.3 vLLM handoff blobs (`glm53-handoff-raw-v1`).

The owner's `Glm53HandoffConnector` already exports a request's prefill state from the live vLLM
server. For Drift only the per-token MLA latents of the sparse-attention layers are read: they
are exactly GLM's canonical entries (`kv_a_layernorm(kv_a_proj(x))`, NoPE). The blob also carries
prompt token ids, selector caches, recurrent state pages and hidden-state captures; none of those
are forwarded on the Drift channel, and this module never returns the token ids.

Cache format `fp8_ds_mla` (per token, 656 bytes): 512 x float8_e4m3fn latent, 4 x float32 scales
(one per 128 channels), 64 x bfloat16 rope slot (unused by this NoPE model). NumPy only.
"""
from __future__ import annotations
import json
import re
import struct
from pathlib import Path
import numpy as np

MAGIC = b"G53HAND1"
_ATTN = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.attn$")


def _e4m3fn_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent, mantissa = (byte >> 3) & 0xF, byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            value = np.nan                                   # the only NaN encoding; e4m3fn has no inf
        elif exponent == 0:
            value = (mantissa / 8.0) * 2.0 ** -6             # subnormal
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = sign * value
    return table


E4M3FN = _e4m3fn_table()


def read_header(path: Path) -> tuple[dict, int]:
    with Path(path).open("rb") as handle:
        if handle.read(8) != MAGIC:
            raise ValueError("not a glm53 handoff blob")
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    data_start = -(-(16 + length) // 4096) * 4096            # header is zero-padded to 4096
    return header, data_start


def dequantize_fp8_ds_mla(rows: np.ndarray, latent_dim: int = 512, group: int = 128) -> np.ndarray:
    """rows: uint8 [T, 656] -> float32 [T, latent_dim]."""
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] < latent_dim + 4 * (latent_dim // group):
        raise ValueError("unexpected fp8_ds_mla row shape")
    groups = latent_dim // group
    values = E4M3FN[rows[:, :latent_dim]]
    scales = rows[:, latent_dim:latent_dim + 4 * groups].copy().view("<f4")          # [T, groups]
    out = values.reshape(-1, groups, group) * scales[:, :, None]
    if not np.isfinite(out).all():
        raise ValueError("nonfinite latent after dequantization")
    return out.reshape(-1, latent_dim).astype(np.float32)


_POSITIVE = np.array([E4M3FN[b] for b in range(0x7F)], dtype=np.float32)          # codes 0x00..0x7E ascending: 0 .. 448


def pack_fp8_ds_mla(latents: np.ndarray, group: int = 128) -> np.ndarray:
    """float [T, 512] -> uint8 [T, 512 + 4*groups]: e4m3fn codes then one float32 scale per group.
    The inverse of `dequantize_fp8_ds_mla` up to fp8 rounding. The rope slot is not produced (NoPE model)."""
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % group or not np.isfinite(x).all():
        raise ValueError("latents must be finite [T, multiple of the group size]")
    tiles = x.reshape(x.shape[0], -1, group)
    scales = np.maximum(np.abs(tiles).max(-1, keepdims=True) / 448.0, 1e-12).astype(np.float32)
    scaled = np.clip(np.abs(tiles) / scales, 0.0, 448.0)
    upper = np.clip(np.searchsorted(_POSITIVE, scaled), 1, len(_POSITIVE) - 1)
    lower = upper - 1
    codes = np.where(scaled - _POSITIVE[lower] <= _POSITIVE[upper] - scaled, lower, upper).astype(np.uint8)
    codes |= (np.signbit(tiles).astype(np.uint8) << 7)
    return np.concatenate((codes.reshape(x.shape[0], -1), scales.reshape(x.shape[0], -1).view(np.uint8)), axis=1)


def read_header_bytes(blob) -> tuple[dict, int]:
    """Same as read_header for a blob already in memory (e.g. the shared-memory window an RDMA pull landed in)."""
    view = memoryview(blob)
    if bytes(view[:8]) != MAGIC:
        raise ValueError("not a glm53 handoff blob")
    (length,) = struct.unpack("<Q", view[8:16])
    if 16 + length > len(view):
        raise ValueError("truncated glm53 handoff header")
    return json.loads(bytes(view[16:16 + length])), -(-(16 + length) // 4096) * 4096


def read_latents(path, skip_draft_layers: bool = True) -> dict[int, np.ndarray]:
    """Per sparse-attention layer: float32 [n_tokens, 512] canonical latents, in position order. `path` may be a file or a bytes-like blob."""
    in_memory = not isinstance(path, (str, Path))
    header, data_start = read_header_bytes(path) if in_memory else read_header(path)
    if header.get("format") != "glm53-handoff-raw-v1":
        raise ValueError(f"unsupported blob format {header.get('format')}")
    if header.get("cache_config", {}).get("cache_dtype") != "fp8_ds_mla":
        raise ValueError("this reader is qualified for fp8_ds_mla caches only")
    if int(header.get("export_from", 0)) != 0:
        raise ValueError("delta exports are not supported; request a full export")
    n_tokens = int(header["n_tokens"])
    raw = np.frombuffer(path, dtype=np.uint8)[data_start:] if in_memory else np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)
    out: dict[int, np.ndarray] = {}
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
        if pages * slots < n_tokens:
            raise ValueError("exported pages do not cover the prompt")
        block = np.frombuffer(raw[tensor["offset"]: tensor["offset"] + tensor["nbytes"]], dtype=np.uint8).reshape(pages * slots, width)
        out[int(match.group(1))] = dequantize_fp8_ds_mla(np.ascontiguousarray(block[:n_tokens]))
    if not out:
        raise ValueError("no MLA attention layers found in the blob")
    return dict(sorted(out.items()))


def summary(path: Path) -> dict:
    header, _ = read_header(path)
    return {"format": header["format"], "n_tokens": header["n_tokens"], "tp_rank": header["tp_rank"], "tp_size": header["tp_size"],
            "cache_dtype": header.get("cache_config", {}).get("cache_dtype"), "tensors": len(header["tensors"])}
