"""Serving-stack connector core (docs/reference/service/SERVING_INTEGRATION.md). No vLLM imports here.

Pure tensor logic a KV-connector shim calls:

  tap:     native paged-cache rows for a request  -> canonical entries (de-rotated K, V; or MLA latent)
  inject:  canonical entries for N placeholder tokens -> native paged-cache rows at their slots
  plan:    how many leading prompt tokens are Drift placeholders for this request

Paged layouts differ by attention backend, so the layout is declared per layer:
  "kv_first"     [2, num_blocks, block_size, H, D]   (FlashAttention-style)
  "blocks_first" [num_blocks, 2, block_size, H, D]   (FlashInfer-style)
  "mla"          [num_blocks, block_size, C]          (latent cache, unquantized)
Quantized caches (e.g. fp8_ds_mla) are refused until their packing is qualified.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
from drift.core.types import KV

LAYOUTS = {"kv_first", "blocks_first", "mla"}


@dataclass(frozen=True)
class RopeSpec:
    """Split-half (NeoX-style) rotary on the first `rotary_dim` dims; text-only positions."""
    theta: float
    rotary_dim: int

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=positions.device) / self.rotary_dim))
        phase = positions.float()[:, None] * inv[None, :]
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos(), phase.sin()

    def _apply(self, x: torch.Tensor, positions: torch.Tensor, sign: float) -> torch.Tensor:
        """x: [T, H, D]. sign=+1 rotates, sign=-1 inverts the rotation exactly."""
        cos, sin = self.cos_sin(positions)
        cos, sin = cos[:, None, :].to(x.dtype), (sign * sin)[:, None, :].to(x.dtype)
        rot, rest = x[..., : self.rotary_dim], x[..., self.rotary_dim:]
        half = self.rotary_dim // 2
        turned = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
        return torch.cat((rot * cos + turned * sin, rest), dim=-1)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self._apply(x, positions, +1.0)

    def derotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self._apply(x, positions, -1.0)


@dataclass(frozen=True)
class LayerSpec:
    name: str                      # the server's layer name, e.g. "model.layers.3.self_attn.attn"
    model_layer: int               # adapter/descriptor layer index
    layout: str
    rope: RopeSpec | None = None   # None for NoPE / MLA latents without a rope part
    cache_dtype: str = "auto"

    def check(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError("unknown paged layout")
        if self.cache_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(f"cache dtype {self.cache_dtype} is not qualified; run the server with an unquantized KV cache")
        if self.layout == "mla" and self.rope is not None:
            raise ValueError("MLA latents with a rope part are not qualified yet")


def _rows(cache: torch.Tensor, layout: str, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather rows for flat slot ids. Returns (k_or_latent, v_or_None)."""
    if layout == "mla":
        blocks, size, width = cache.shape
        return cache.reshape(blocks * size, width)[slots], None
    if layout == "kv_first":
        _, blocks, size, heads, dim = cache.shape
        flat = cache.reshape(2, blocks * size, heads, dim)
        return flat[0, slots], flat[1, slots]
    blocks, _, size, heads, dim = cache.shape
    block, offset = slots // size, slots % size
    return cache[block, 0, offset], cache[block, 1, offset]


def tap_layer(spec: LayerSpec, cache: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor) -> KV | torch.Tensor:
    """Canonical entries for the tokens at `slots` (their sequence `positions` given)."""
    spec.check()
    if slots.ndim != 1 or slots.shape != positions.shape or slots.numel() == 0:
        raise ValueError("slots and positions must be equal-length nonempty vectors")
    k, v = _rows(cache, spec.layout, slots.to(cache.device))
    if spec.layout == "mla":
        return k.clone()
    k = spec.rope.derotate(k.float(), positions.to(k.device)).to(k.dtype) if spec.rope is not None else k.clone()
    entry = KV(k, v.clone())
    entry.check()
    return entry


def inject_layer(spec: LayerSpec, cache: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor,
                 entry: KV | torch.Tensor) -> None:
    """Write canonical entries into the native cache rows at `slots` (in place, like the server does)."""
    spec.check()
    slots = slots.to(cache.device)
    if spec.layout == "mla":
        if not isinstance(entry, torch.Tensor) or entry.ndim != 2 or entry.shape != (slots.numel(), cache.shape[-1]):
            raise ValueError("MLA injection expects [N, C] latents matching the cache width")
        if not torch.isfinite(entry).all():
            raise ValueError("nonfinite latent")
        blocks, size, width = cache.shape
        cache.view(blocks * size, width)[slots] = entry.to(cache)
        return
    if not isinstance(entry, KV):
        raise ValueError("split layouts expect KV entries")
    entry.check()
    if entry.tokens != slots.numel():
        raise ValueError("entry count must equal the placeholder slots")
    k = spec.rope.rotate(entry.k.float(), positions.to(entry.k.device)).to(cache.dtype) if spec.rope is not None else entry.k.to(cache.dtype)
    v = entry.v.to(cache.dtype)
    if spec.layout == "kv_first":
        _, blocks, size, heads, dim = cache.shape
        if k.shape[1:] != (heads, dim):
            raise ValueError("entry head shape does not match this rank's cache (check tensor-parallel sharding)")
        flat = cache.view(2, blocks * size, heads, dim)
        flat[0, slots], flat[1, slots] = k.to(cache.device), v.to(cache.device)
        return
    blocks, _, size, heads, dim = cache.shape
    if k.shape[1:] != (heads, dim):
        raise ValueError("entry head shape does not match this rank's cache (check tensor-parallel sharding)")
    block, offset = slots // size, slots % size
    cache[block, 0, offset], cache[block, 1, offset] = k.to(cache.device), v.to(cache.device)


def shard_heads(entry: KV, rank: int, world: int) -> KV:
    """Tensor-parallel servers shard KV heads across ranks; a rank injects only its slice."""
    heads = entry.k.shape[1]
    if world <= 0 or not 0 <= rank < world or heads % world:
        raise ValueError("KV heads must divide evenly across tensor-parallel ranks")
    step = heads // world
    return KV(entry.k[:, rank * step:(rank + 1) * step], entry.v[:, rank * step:(rank + 1) * step])


@dataclass(frozen=True)
class PlaceholderPlan:
    placeholders: int            # leading prompt tokens that are Drift placeholders
    matched: int                 # how many of them the connector can fill (0 = exact stock request)


def plan_request(prompt_token_ids: Sequence[int], placeholder_id: int, available_entries: int,
                 already_computed: int = 0, block_size: int = 1) -> PlaceholderPlan:
    """Leading run of `placeholder_id` marks where foreign memory goes. The connector claims
    them only if it holds exactly that many entries; a mismatch is refused, never padded.
    Servers account external tokens in whole cache blocks, so the entry count must be a
    multiple of `block_size`; the sender's window policy trims to a multiple instead of padding."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    if available_entries % block_size:
        raise ValueError(f"{available_entries} entries is not a multiple of the server block size {block_size}")
    run = 0
    for token in prompt_token_ids:
        if token != placeholder_id:
            break
        run += 1
    if run == 0 or available_entries == 0:
        return PlaceholderPlan(run, 0)
    if available_entries != run:
        raise ValueError(f"request reserves {run} placeholder tokens but the session holds {available_entries} entries")
    if len(prompt_token_ids) == run:
        raise ValueError("a request needs at least one real token after the placeholders")
    return PlaceholderPlan(run, max(0, run - already_computed))


def placeholder_prompt(placeholder_id: int, entries: int, real_prompt: Sequence[int]) -> list[int]:
    if entries < 0 or not real_prompt:
        raise ValueError("need a real prompt and a nonnegative entry count")
    return [placeholder_id] * entries + list(real_prompt)


def tap_request(specs: Sequence[LayerSpec], caches: Mapping[str, torch.Tensor], slots: torch.Tensor,
                positions: torch.Tensor) -> dict[int, KV | torch.Tensor]:
    return {spec.model_layer: tap_layer(spec, caches[spec.name], slots, positions) for spec in specs}


def inject_request(specs: Sequence[LayerSpec], caches: Mapping[str, torch.Tensor], slots: torch.Tensor,
                   entries: Mapping[int, KV | torch.Tensor]) -> None:
    """Foreign memory occupies sequence positions 0..N-1 of the receiving request."""
    positions = torch.arange(slots.numel())
    missing = [spec.model_layer for spec in specs if spec.model_layer not in entries]
    if missing:
        raise ValueError(f"no entries for KV-bearing layers {missing}; injection must be complete across layers")
    for spec in specs:
        inject_layer(spec, caches[spec.name], slots, positions, entries[spec.model_layer])
