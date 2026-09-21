"""M4.1 MLX adapter qualification: tiny random configs, torch weights transferred to
mlx-vlm 0.7.1, cross-runtime parity against transformers 5.17.0.

Mirrors `tests/test_adapters_next.py` for `MlxQwen4ExpAdapter` / `MlxGlm5NextAdapter`,
plus stock parity of the MLX port against the torch reference. Runs in `.venv-next`;
skipped elsewhere. Evidence and every decision: docs/research/M4_MLX_NOTES.md.

Precision preregistration:
- MLX-vs-MLX checks (adapter vs stock MLX forward, incremental decode, identity
  import, foreign-path parity across runtimes) use the D19 FP32 tolerance 2e-5.
- Cross-runtime stock parity was preregistered at atol/rtol 1e-4 and tightened to
  CROSS_TOL after it passed (observed max abs 3.4e-8 Qwen, 7.2e-7 GLM).
- Metal fp32 GEMM on Apple M5 (applegpu_g17) is TF32-class in mlx 0.32.2 (operands
  rounded to 10-bit mantissas; measured, see notes). The override below selects the
  full-precision kernels; `test_metal_fp32_gemm_is_full_precision` verifies it took
  effect in this process. Without it the cross-runtime checks fail at 1e-4 (Qwen 1.6e-4,
  GLM 6.6e-3 max abs) and prefill/decode consistency fails at 2e-5.
"""
from __future__ import annotations
import os
os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g16s")   # must precede the first Metal use

import pytest
import torch

hf = pytest.importorskip("transformers")
mx = pytest.importorskip("mlx.core")
mlx_vlm = pytest.importorskip("mlx_vlm")
if hf.__version__ != "5.17.0" or mlx_vlm.__version__ != "0.7.1":
    pytest.skip("MLX adapters are qualified on transformers==5.17.0 and mlx-vlm==0.7.1", allow_module_level=True)

from drift.adapters import mlx_fixtures as fx
from drift.adapters.base import ForeignEntries
from drift.core.attention import Gate
from drift.core.types import KV

TOL = dict(atol=2e-5, rtol=2e-5)          # D19
CROSS_TOL = dict(atol=2e-6, rtol=2e-6)    # preregistered 1e-4, tightened after passing
FAMILIES = list(fx.FAMILIES)


def hostile_foreign(adapter, tokens=3):
    d = adapter.descriptor
    layers = {}
    for i in d.kv_layers:
        if d.layout == "kv_split":
            layers[i] = KV(torch.full((tokens, d.kv_heads, d.head_dim), 1e3),
                           torch.full((tokens, d.kv_heads, d.head_dim), -1e3))
        else:
            layers[i] = torch.full((tokens, d.head_dim), 1e3)
    return ForeignEntries(torch.arange(tokens), layers)


def mlx_stock(family, model, ids):
    return fx.mlx_stock_forward(family, model, ids)[0]


def native_keys(adapter, state, layer):
    cache = adapter._kv_cache(state, layer)
    return fx.to_torch(cache.keys[..., :cache.offset, :]), fx.to_torch(cache.values[..., :cache.offset, :])


# -- runtime precision --------------------------------------------------------------------

def test_metal_fp32_gemm_is_full_precision():
    """mlx 0.32.2 on applegpu_g17 rounds fp32 GEMM operands to TF32 unless the arch
    override is honoured. The suite's tolerances assume true fp32."""
    p = fx.gemm_precision()
    assert p["vs_fp64"] < 1e-6, (
        f"fp32 GEMM error {p['vs_fp64']:.2e} vs fp64 (vs TF32-rounded inputs {p['vs_tf32_inputs']:.2e}, "
        f"arch {p['architecture']}): MLX_METAL_GPU_ARCH override not in effect; MLX was initialised before this "
        "module set it, or the override no longer selects the full-precision kernels")


# -- cross-runtime stock parity -----------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [1, 2, 7, 16])
def test_cross_runtime_stock_parity(family, n):
    """MLX stock forward == torch stock forward (float32 both sides), full prefill and a
    cached 5-token continuation. Positions at or after an exact tie in a sparse
    indexer's top-k cut are excluded and reported: torch.topk and mx.argpartition break
    ties differently, and a flipped selection changes every later layer and position."""
    torch_adapter, mlx_adapter, torch_model, mlx_model = fx.build_adapters(family)
    ids = torch.randint(3, 97, (n + 5,))
    out = fx.exchange(family, torch_model, mlx_model, ids, split=n)
    ties = fx.selection_ties(family, mlx_adapter, ids)
    limit = fx.first_tie(ties)
    limit = len(ids) if limit is None else limit
    errors = {k: fx.max_error(out[f"mlx_{k}"], out[f"torch_{k}"]) for k in ("full", "cont")}
    note = f"family={family} n={n} ties={ties} errors={errors}"
    assert limit >= min(11, len(ids)), note         # a tie needs three complete blocks/pools
    # Positions before the first tie must agree on both runtimes.
    torch.testing.assert_close(out["mlx_full"][:limit], out["torch_full"][:limit], **CROSS_TOL, msg=note)
    torch.testing.assert_close(out["mlx_cont"][:max(0, limit - n)], out["torch_cont"][:max(0, limit - n)], **CROSS_TOL, msg=note)
    # Positions from the first tie on agree unless the runtimes broke the tie differently.
    tail_agrees = (torch.allclose(out["mlx_full"][limit:], out["torch_full"][limit:], **CROSS_TOL)
                   and torch.allclose(out["mlx_cont"][max(0, limit - n):], out["torch_cont"][max(0, limit - n):], **CROSS_TOL))
    if not tail_agrees:
        assert ties, note
        pytest.xfail(f"exact sparse-indexer tie broken differently: positions >= {limit} differ ({note})")


# -- qualification suite (mirrors test_adapters_next.py) -------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [1, 2, 7, 16])
def test_stock_parity_incremental_and_hard_off(family, n):
    _, adapter, _, model = fx.build_adapters(family)
    ids = torch.randint(3, 97, (n + 5,))
    before = adapter.frozen_digest()
    reference = mlx_stock(family, model, ids)
    full = adapter.forward(ids)
    torch.testing.assert_close(full.output, reference, **TOL)
    assert set(full.canonical) == set(adapter.descriptor.kv_layers)

    prefix = adapter.forward(ids[:n])
    continued = adapter.forward(ids[n:], prefix.state)
    torch.testing.assert_close(continued.output, reference[n:], **TOL)

    state, steps = None, []
    for token in ids:
        out = adapter.forward(token[None], state)
        state, steps = out.state, steps + [out.output]
    torch.testing.assert_close(torch.cat(steps), reference, **TOL)

    # The input state must survive a forward unchanged (deep copy).
    again = adapter.forward(ids[n:], prefix.state)
    assert torch.equal(again.output, continued.output)

    hostile = hostile_foreign(adapter)
    closed = adapter.forward(ids[n:], prefix.state, foreign=hostile, override=0.0)
    assert torch.equal(closed.output, continued.output)
    for ours, theirs in zip(adapter.cache_tensors(closed.state), adapter.cache_tensors(continued.state)):
        assert torch.equal(ours, theirs)
    assert not closed.foreign_mass

    opened = adapter.forward(ids[n:], prefix.state, foreign=hostile, override=1.0)
    assert not torch.allclose(opened.output, continued.output)
    for layer in adapter.descriptor.kv_layers:
        mass = opened.foreign_mass[layer]
        assert mass.shape[0] == len(ids) - n and bool((mass >= 0).all()) and bool((mass <= 1).all())
        assert opened.entry_mass[layer].shape == (3,)
    assert adapter.frozen_digest() == before


@pytest.mark.parametrize("family", FAMILIES)
def test_capture_boundary_matches_native_cache(family):
    _, adapter, _, _ = fx.build_adapters(family)
    ids = torch.randint(3, 97, (9,))
    out = adapter.forward(ids)
    for i in adapter.descriptor.kv_layers:
        keys, values = native_keys(adapter, out.state, i)
        if family == "qwen4_exp":
            kv = out.canonical[i]
            assert kv.k.shape == (9, adapter.descriptor.kv_heads, adapter.descriptor.head_dim)
            k, v = adapter.native_kv_from_canonical(i, kv, torch.arange(9))
            torch.testing.assert_close(fx.to_torch(k), keys, **TOL)          # pre-rotary + model rotary == cache
            torch.testing.assert_close(fx.to_torch(v), values, **TOL)        # values unrotated
            assert not torch.allclose(kv.k.transpose(0, 1)[None], keys)      # the canonical K is NOT rotated
        else:
            latent = out.canonical[i]
            assert latent.shape == (9, adapter.descriptor.head_dim)
            torch.testing.assert_close(latent[None, None], keys, **TOL)      # the MLX cache is the latent
            assert values.shape[-1] == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_self_entries_as_foreign_are_attended(family):
    """Own canonical entries offered back as foreign memory must be attendable through a
    gate (consumed as a float on MLX: no gradient path into torch)."""
    _, adapter, _, _ = fx.build_adapters(family)
    source = torch.randint(3, 97, (6,))
    captured = adapter.forward(source)
    foreign = ForeignEntries(torch.arange(6), {i: captured.canonical[i] for i in adapter.descriptor.kv_layers})
    gates = {i: Gate(-1.0) for i in adapter.descriptor.kv_layers}
    query = torch.randint(3, 97, (4,))
    out = adapter.forward(query, foreign=foreign, gates=gates)
    native = adapter.forward(query)
    assert not torch.allclose(out.output, native.output)
    for i in adapter.descriptor.kv_layers:
        assert float(out.foreign_mass[i].mean()) > 0
        assert out.entry_mass[i].shape == (6,) and float(out.entry_mass[i].sum()) > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_rejections(family):
    _, adapter, _, _ = fx.build_adapters(family)
    non_kv = next(i for i in range(4) if i not in adapter.descriptor.kv_layers)
    bad = ForeignEntries(torch.arange(1), {non_kv: torch.zeros(1, 4)})
    with pytest.raises(ValueError, match="non-KV"):
        adapter.forward(torch.tensor([3, 4]), foreign=bad, override=1.0)
    hostile = hostile_foreign(adapter)
    with pytest.raises(ValueError):
        adapter.forward(torch.tensor([3, 4]), foreign=hostile)          # no gate, no override
    with pytest.raises(ValueError):
        adapter.forward(torch.tensor([3, 4]), foreign=hostile, override=1.5)
    with pytest.raises(ValueError):
        adapter.forward(torch.tensor([3, 4], dtype=torch.int32))
    with pytest.raises(ValueError):
        adapter.forward(torch.tensor([], dtype=torch.long))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [1, 5, 12])
def test_identity_replacement_and_state_snapshot(family, n):
    _, adapter, _, _ = fx.build_adapters(family)
    ids = torch.randint(3, 97, (n + 4,))
    prefix = adapter.forward(ids[:n])
    ordinary = adapter.forward(ids[n:], prefix.state)
    imported = adapter.import_self_prefix(prefix.state, prefix.canonical, torch.arange(n))
    transplant = adapter.forward(ids[n:], imported)
    torch.testing.assert_close(transplant.output, ordinary.output, **TOL)
    for i in adapter.descriptor.kv_layers:
        for ours, theirs in zip(native_keys(adapter, imported, i), native_keys(adapter, prefix.state, i)):
            torch.testing.assert_close(ours, theirs, **TOL)
    snapshot = adapter.snapshot_state(prefix.state)
    restored = adapter.restore_state(prefix.state, snapshot)
    resumed = adapter.forward(ids[n:], restored)
    assert torch.equal(resumed.output, ordinary.output)
    for ours, theirs in zip(adapter.cache_tensors(restored), adapter.cache_tensors(prefix.state)):
        assert torch.equal(ours, theirs)
    with pytest.raises(ValueError):
        adapter.restore_state(prefix.state, {"tensors": snapshot["tensors"], "skeleton": {**snapshot["skeleton"], "layer99": {}}})
    with pytest.raises(ValueError):
        adapter.import_self_prefix(prefix.state, {}, torch.arange(n))
    with pytest.raises(ValueError):
        adapter.import_self_prefix(prefix.state, prefix.canonical, torch.arange(1, n + 1))


@pytest.mark.parametrize("family", FAMILIES)
def test_embeddings_path_matches_ids(family):
    _, adapter, _, _ = fx.build_adapters(family)
    ids = torch.randint(3, 97, (6,))
    by_ids = adapter.forward(ids)
    by_embeds = adapter.forward_embeddings(adapter.embed(ids))
    torch.testing.assert_close(by_embeds.output, by_ids.output, **TOL)


# -- cross-runtime foreign path ------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("override", [1.0, 0.5])
def test_cross_runtime_foreign_path_parity(family, override):
    """The same foreign entries fed to the torch adapter and the MLX adapter give the same
    output and foreign mass. Source entries come from the torch adapter; a second run uses
    the MLX capture so the canonical entries themselves are compared too."""
    torch_adapter, mlx_adapter, _, _ = fx.build_adapters(family)
    source = torch.randint(3, 97, (6,))
    query = torch.randint(3, 97, (4,))
    with torch.no_grad():
        t_src = torch_adapter.forward(source)
    m_src = mlx_adapter.forward(source)
    for i in torch_adapter.descriptor.kv_layers:
        a, b = t_src.canonical[i], m_src.canonical[i]
        if isinstance(a, KV):
            torch.testing.assert_close(b.k, a.k, **CROSS_TOL)
            torch.testing.assert_close(b.v, a.v, **CROSS_TOL)
        else:
            torch.testing.assert_close(b, a, **CROSS_TOL)
    foreign = ForeignEntries(torch.arange(6), {i: t_src.canonical[i] for i in torch_adapter.descriptor.kv_layers})
    with torch.no_grad():
        t_out = torch_adapter.forward(query, foreign=foreign, override=override)
        t_native = torch_adapter.forward(query)
    m_out = mlx_adapter.forward(query, foreign=foreign, override=override)
    torch.testing.assert_close(m_out.output, t_out.output, **TOL)
    assert not torch.allclose(t_out.output, t_native.output)
    for i in torch_adapter.descriptor.kv_layers:
        torch.testing.assert_close(m_out.foreign_mass[i], t_out.foreign_mass[i], **TOL)
        torch.testing.assert_close(m_out.entry_mass[i], t_out.entry_mass[i], **TOL)
    # Prefix state on both sides, then the same foreign block on a continuation.
    with torch.no_grad():
        t_prefix = torch_adapter.forward(query[:2])
        t_cont = torch_adapter.forward(query[2:], t_prefix.state, foreign=foreign, override=override)
    m_prefix = mlx_adapter.forward(query[:2])
    m_cont = mlx_adapter.forward(query[2:], m_prefix.state, foreign=foreign, override=override)
    torch.testing.assert_close(m_cont.output, t_cont.output, **TOL)
