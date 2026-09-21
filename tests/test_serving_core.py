"""Connector core on mocked paged caches: tap/inject for every layout, rope inverse, planning, store."""
from __future__ import annotations
from uuid import UUID
import pytest
import torch
from drift.core.types import KV
from drift.serving.core import (LayerSpec, RopeSpec, inject_layer, inject_request, placeholder_prompt,
                                    plan_request, shard_heads, tap_layer, tap_request)
from drift.serving.store import SessionStore
from drift.transport.wire2 import FrameCodec2, Publication

ROPE = RopeSpec(theta=10000.0, rotary_dim=4)


def paged(layout, blocks=6, size=4, heads=2, dim=8, width=16):
    torch.manual_seed(0)
    if layout == "mla":
        return torch.randn(blocks, size, width)
    if layout == "kv_first":
        return torch.randn(2, blocks, size, heads, dim)
    return torch.randn(blocks, 2, size, heads, dim)


def test_rope_is_exactly_invertible_and_partial():
    x = torch.randn(5, 2, 8)
    positions = torch.tensor([0, 3, 7, 100, 4096])
    rotated = ROPE.rotate(x, positions)
    assert torch.equal(rotated[..., 4:], x[..., 4:])                  # only the first rotary_dim dims move
    assert torch.equal(rotated[0], x[0])                              # position 0 is the identity
    torch.testing.assert_close(ROPE.derotate(rotated, positions), x, rtol=1e-5, atol=1e-5)
    # relative-position property: <rot(q,m), rot(k,n)> depends on m-n only
    q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    a = (ROPE.rotate(q, torch.tensor([9])) * ROPE.rotate(k, torch.tensor([4]))).sum()
    b = (ROPE.rotate(q, torch.tensor([105])) * ROPE.rotate(k, torch.tensor([100]))).sum()
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("layout", ["kv_first", "blocks_first"])
def test_inject_then_tap_roundtrips_canonical_entries(layout):
    cache = paged(layout)
    spec = LayerSpec("layers.3.attn", 3, layout, ROPE)
    slots = torch.tensor([9, 2, 17, 4])                               # scattered pages, like a real block table
    positions = torch.arange(4)
    entry = KV(torch.randn(4, 2, 8), torch.randn(4, 2, 8))
    untouched = cache.clone()
    inject_layer(spec, cache, slots, positions, entry)
    back = tap_layer(spec, cache, slots, positions)
    torch.testing.assert_close(back.k, entry.k, rtol=1e-5, atol=1e-5)
    assert torch.equal(back.v, entry.v)
    # every other slot is untouched
    other = torch.tensor([s for s in range(24) if s not in slots.tolist()])
    before = tap_layer(LayerSpec("x", 0, layout, None), untouched, other, torch.zeros_like(other))
    after = tap_layer(LayerSpec("x", 0, layout, None), cache, other, torch.zeros_like(other))
    assert torch.equal(before.k, after.k) and torch.equal(before.v, after.v)
    # the native rows really are rotated (the server's format), not canonical
    raw = tap_layer(LayerSpec("x", 0, layout, None), cache, slots, positions)
    assert not torch.allclose(raw.k[1:], entry.k[1:])


def test_mla_latents_roundtrip_and_quantized_caches_are_refused():
    cache = paged("mla")
    spec = LayerSpec("layers.7.attn", 7, "mla")
    slots = torch.tensor([1, 22, 8])
    latents = torch.randn(3, 16)
    inject_layer(spec, cache, slots, torch.arange(3), latents)
    assert torch.equal(tap_layer(spec, cache, slots, torch.arange(3)), latents)
    with pytest.raises(ValueError, match="not qualified"):
        tap_layer(LayerSpec("l", 7, "mla", cache_dtype="fp8_ds_mla"), cache, slots, torch.arange(3))
    with pytest.raises(ValueError):
        inject_layer(spec, cache, slots, torch.arange(3), torch.randn(3, 15))
    with pytest.raises(ValueError):
        inject_layer(spec, cache, slots, torch.arange(3), KV(torch.randn(3, 1, 16), torch.randn(3, 1, 16)))


def test_request_level_tap_and_inject_require_every_kv_layer():
    specs = [LayerSpec("a", 3, "kv_first", ROPE), LayerSpec("b", 7, "mla")]
    caches = {"a": paged("kv_first"), "b": paged("mla")}
    slots = torch.tensor([5, 6, 7])
    entries = {3: KV(torch.randn(3, 2, 8), torch.randn(3, 2, 8)), 7: torch.randn(3, 16)}
    inject_request(specs, caches, slots, entries)
    tapped = tap_request(specs, caches, slots, torch.arange(3))
    torch.testing.assert_close(tapped[3].k, entries[3].k, rtol=1e-5, atol=1e-5)
    assert torch.equal(tapped[7], entries[7])
    with pytest.raises(ValueError, match="complete across layers"):
        inject_request(specs, caches, slots, {3: entries[3]})


def test_tensor_parallel_head_sharding():
    entry = KV(torch.randn(3, 4, 8), torch.randn(3, 4, 8))
    parts = [shard_heads(entry, r, 2) for r in range(2)]
    assert torch.equal(torch.cat([p.k for p in parts], dim=1), entry.k)
    with pytest.raises(ValueError):
        shard_heads(entry, 0, 3)
    cache = paged("kv_first", heads=2)
    with pytest.raises(ValueError, match="tensor-parallel"):
        inject_layer(LayerSpec("a", 0, "kv_first", None), cache, torch.tensor([0, 1, 2]), torch.arange(3), entry)


def test_placeholder_planning_is_exact_or_stock():
    pad = 7
    prompt = placeholder_prompt(pad, 3, [11, 12])
    assert prompt == [7, 7, 7, 11, 12]
    assert plan_request(prompt, pad, available_entries=3).matched == 3
    assert plan_request(prompt, pad, available_entries=3, already_computed=2).matched == 1
    assert plan_request([11, 12], pad, available_entries=3).matched == 0          # stock request
    assert plan_request(prompt, pad, available_entries=0).matched == 0            # hard-off: nothing injected
    with pytest.raises(ValueError, match="reserves 3"):
        plan_request(prompt, pad, available_entries=2)                            # never padded or truncated
    with pytest.raises(ValueError, match="real token"):
        plan_request([7, 7, 7], pad, available_entries=3)
    with pytest.raises(ValueError, match="block size 16"):
        plan_request(placeholder_prompt(pad, 20, [1]), pad, available_entries=20, block_size=16)
    assert plan_request(placeholder_prompt(pad, 32, [1]), pad, available_entries=32, block_size=16).matched == 32


def test_session_store_is_ordered_atomic_and_authenticated(tmp_path):
    session = UUID(int=5)
    store = SessionStore(tmp_path, FrameCodec2(b"s" * 32))
    pubs = [Publication(session, 1, "ab" * 32, e, e, 2 * e, {0: torch.randn(2, 6)}) for e in range(3)]
    for pub in (pubs[0], pubs[2]):
        store.append(pub)
    assert [p.sequence for p in store.read_from(session, 1, 0)] == [0]            # stops at the gap
    store.append(pubs[1])
    got = store.read_from(session, 1, 0)
    assert [p.sequence for p in got] == [0, 1, 2] and torch.equal(got[2].levels[0], pubs[2].levels[0])
    assert store.read_from(session, 1, 3) == []
    with pytest.raises(FileExistsError):
        store.append(pubs[1])
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ValueError, match="authentication"):
        SessionStore(tmp_path, FrameCodec2(b"x" * 32)).read_from(session, 1, 0)
