"""Level-1 adapter qualification: tiny random configs vs stock transformers 5.17.0.

Architecture parity only (docs/research §9 level 1). No pretrained weights, no
semantics. Runs in the separate `.venv-next` environment; skipped elsewhere.
"""
from __future__ import annotations
import pytest
import torch
from drift.adapters.base import ForeignEntries
from drift.core.attention import Gate
from drift.core.types import KV

hf = pytest.importorskip("transformers")
if hf.__version__ != "5.17.0":
    pytest.skip("adapters are qualified on transformers==5.17.0", allow_module_level=True)

TOL = dict(atol=2e-5, rtol=2e-5)


def make_qwen():
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    from drift.adapters.qwen4_exp import Qwen4ExpAdapter
    torch.manual_seed(0)
    config = Qwen4ExpTextConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0]},
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts_per_tok=2, num_experts=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        hc_count=2, hc_lowrank=8, ple_layer_ids=[], indexer_n_heads=2, indexer_kv_heads=1,
        indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4)
    stock = Qwen4ExpForCausalLM(config).eval()
    stock.config._attn_implementation = "eager"
    adapter = Qwen4ExpAdapter(stock)
    return stock, adapter, lambda ids: stock(input_ids=ids[None], use_cache=False).logits[0]


def make_glm():
    from transformers import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    from drift.adapters.glm5_next import Glm5NextAdapter
    torch.manual_seed(0)
    config = Glm5NextTextConfig(
        vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=4,
        routed_scaling_factor=1.0, kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=0, v_head_dim=8,
        qk_nope_head_dim=8, n_group=1, topk_group=1, num_experts_per_tok=2, max_position_embeddings=256,
        mlp_layer_types=["dense", "sparse", "dense", "sparse"], index_topk=8, index_head_dim=8, index_n_heads=2,
        layer_types=["linear_attention", "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"],
        indexer_types=["full", "full", "full", "full"], linear_head_dim=8, linear_num_heads=4,
        linear_conv_kernel_dim=4, hc_mult=2, index_kpool=4, index_kpool_always_select_tail=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    stock = Glm5NextTextModel(config).eval()
    stock.config._attn_implementation = "eager"
    adapter = Glm5NextAdapter(stock)
    return stock, adapter, lambda ids: stock(input_ids=ids[None], use_cache=False).last_hidden_state[0]


FAMILIES = {"qwen4_exp": make_qwen, "glm5_next": make_glm}


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


def cache_tensors(state):
    out = []
    for layer in state.layers:
        for name in ("keys", "values", "conv_states", "recurrent_states", "indexer_keys"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                out.append(value)
            elif isinstance(value, (list, tuple)):
                out.extend(v for v in value if isinstance(v, torch.Tensor))
    return out


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [1, 2, 7, 16])
def test_stock_parity_incremental_and_hard_off(family, n):
    stock, adapter, stock_forward = FAMILIES[family]()
    ids = torch.randint(3, 97, (n + 5,))
    before = adapter.frozen_digest()
    with torch.no_grad():
        reference = stock_forward(ids)
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

        # The input state must survive a forward unchanged (deep copy), so the same
        # prefix state can seed counterfactual runs.
        again = adapter.forward(ids[n:], prefix.state)
        assert torch.equal(again.output, continued.output)

        hostile = hostile_foreign(adapter)
        closed = adapter.forward(ids[n:], prefix.state, foreign=hostile, override=0.0)
        assert torch.equal(closed.output, continued.output)
        for ours, theirs in zip(cache_tensors(closed.state), cache_tensors(continued.state)):
            assert torch.equal(ours, theirs)
        assert not closed.foreign_mass

        opened = adapter.forward(ids[n:], prefix.state, foreign=hostile, override=1.0)
        assert not torch.allclose(opened.output, continued.output)
        for layer in adapter.descriptor.kv_layers:
            mass = opened.foreign_mass[layer]
            assert mass.shape[0] == len(ids) - n and bool((mass >= 0).all()) and bool((mass <= 1).all())
    assert adapter.frozen_digest() == before


@pytest.mark.parametrize("family", FAMILIES)
def test_capture_boundary_matches_native_cache(family):
    stock, adapter, _ = FAMILIES[family]()
    ids = torch.randint(3, 97, (9,))
    with torch.no_grad():
        out = adapter.forward(ids)
    for i in adapter.descriptor.kv_layers:
        layer = out.state.layers[i]
        attention = adapter._text.layers[i].self_attn
        if family == "qwen4_exp":
            kv = out.canonical[i]
            positions = torch.arange(9).view(1, 1, -1).expand(3, 1, -1)
            cos, sin = adapter._text.rotary_emb(torch.zeros(1, 9, 1), positions)
            from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
            rotated = apply_rotary_pos_emb(kv.k.transpose(0, 1)[None], cos=cos, sin=sin)
            torch.testing.assert_close(rotated, layer.keys, **TOL)
            torch.testing.assert_close(kv.v.transpose(0, 1)[None], layer.values, **TOL)
        else:
            latent = out.canonical[i]
            assert latent.shape == (9, adapter.descriptor.head_dim)
            k, v = attention.expand_kv(latent[None, None], latent.new_empty(1, 1, 9, 0))
            torch.testing.assert_close(k, layer.keys, **TOL)
            torch.testing.assert_close(v, layer.values, **TOL)


@pytest.mark.parametrize("family", FAMILIES)
def test_self_entries_as_foreign_are_attended_and_gate_trains(family):
    """A model's own canonical entries offered back as foreign memory must be attendable,
    and a learned gate must receive gradient through the frozen backbone."""
    stock, adapter, _ = FAMILIES[family]()
    source = torch.randint(3, 97, (6,))
    with torch.no_grad():
        captured = adapter.forward(source)
    foreign = ForeignEntries(torch.arange(6), {i: captured.canonical[i] for i in adapter.descriptor.kv_layers})
    gates = {i: Gate(-1.0) for i in adapter.descriptor.kv_layers}
    query = torch.randint(3, 97, (4,))
    out = adapter.forward(query, foreign=foreign, gates=gates)
    with torch.no_grad():
        native = adapter.forward(query)
    assert not torch.allclose(out.output, native.output)
    for i in adapter.descriptor.kv_layers:
        assert float(out.foreign_mass[i].detach().mean()) > 0
    out.output.square().sum().backward()
    assert all(g.logit.grad is not None and g.logit.grad.abs() > 0 for g in gates.values())
    assert all(p.grad is None for p in stock.parameters())


@pytest.mark.parametrize("family", FAMILIES)
def test_rejections(family):
    _, adapter, _ = FAMILIES[family]()
    non_kv = next(i for i in range(4) if i not in adapter.descriptor.kv_layers)
    bad = ForeignEntries(torch.arange(1), {non_kv: torch.zeros(1, 4)})
    with pytest.raises(ValueError, match="non-KV"):
        adapter.forward(torch.tensor([3, 4]), foreign=bad, override=1.0)
    hostile = hostile_foreign(adapter)
    with pytest.raises(ValueError):
        adapter.forward(torch.tensor([3, 4]), foreign=hostile)          # no gate, no override
    with pytest.raises(ValueError):
        adapter.forward(torch.tensor([3, 4]), foreign=hostile, override=1.5)


def test_qwen_writes_pool_and_glm_reads_it_through_its_adapter():
    """End-to-end pipe across two architectures: Qwen canonical -> writer -> pool ->
    GLM reader -> GLM attends to it. Random weights, so only the mechanics are asserted."""
    from drift.translate.pool import Layout, Translator, fit_member, fit_pool_format
    _, qwen, _ = make_qwen()
    _, glm, _ = make_glm()
    torch.manual_seed(1)
    ids = torch.randint(3, 97, (48,))
    with torch.no_grad():
        q_out, g_out = qwen.forward(ids), glm.forward(ids)
    qd, gd = qwen.descriptor, glm.descriptor
    q_layout = Layout("kv_split", qd.kv_heads, qd.head_dim)
    g_layout = Layout("mla_latent", 1, gd.head_dim)
    levels = min(len(qd.kv_layers), len(gd.kv_layers))
    q_rows = {lvl: q_layout.flatten(q_out.canonical[qd.kv_layers[lvl]]) for lvl in range(levels)}
    g_rows = {lvl: g_layout.flatten(g_out.canonical[gd.kv_layers[lvl]]) for lvl in range(levels)}
    fmt, pool_rows = fit_pool_format({lvl: {"qwen": q_rows[lvl], "glm": g_rows[lvl]} for lvl in range(levels)}, width=8)
    q_tr = Translator("qwen", q_layout, {qd.kv_layers[lvl]: lvl for lvl in range(levels)}, fmt)
    g_tr = Translator("glm", g_layout, {gd.kv_layers[lvl]: lvl for lvl in range(levels)}, fmt)
    fit_member(q_tr, q_rows, pool_rows)
    fit_member(g_tr, g_rows, pool_rows)
    with torch.no_grad():
        source = qwen.forward(torch.randint(3, 97, (6,)))
        pool = q_tr.write(source.canonical)
        foreign = ForeignEntries(torch.arange(6), g_tr.read(pool))
        query = torch.randint(3, 97, (4,))
        native = glm.forward(query)
        coupled = glm.forward(query, foreign=foreign, override=1.0)
        closed = glm.forward(query, foreign=foreign, override=0.0)
    assert set(foreign.layers) == set(gd.kv_layers)
    assert all(v.shape == (6, gd.head_dim) for v in foreign.layers.values())
    assert not torch.allclose(coupled.output, native.output)
    assert torch.equal(closed.output, native.output)
    assert all(float(coupled.foreign_mass[i].mean()) > 0 for i in gd.kv_layers)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [1, 5, 12])
def test_identity_replacement_and_state_snapshot(family, n):
    """M0.2: KV layers rebuilt from canonical entries at exact positions, recurrent state
    from the reference, must continue exactly like ordinary decoding. Snapshot/restore
    of the whole private state must be lossless."""
    stock, adapter, _ = FAMILIES[family]()
    ids = torch.randint(3, 97, (n + 4,))
    with torch.no_grad():
        prefix = adapter.forward(ids[:n])
        ordinary = adapter.forward(ids[n:], prefix.state)
        # Canonical entries for the whole prefix, from a full-prefix capture.
        imported = adapter.import_self_prefix(prefix.state, prefix.canonical, torch.arange(n))
        transplant = adapter.forward(ids[n:], imported)
        torch.testing.assert_close(transplant.output, ordinary.output, **TOL)
        # Rebuilt KV must equal the native cache tensors, not merely produce similar logits.
        for i in adapter.descriptor.kv_layers:
            torch.testing.assert_close(imported.layers[i].keys, prefix.state.layers[i].keys, **TOL)
            torch.testing.assert_close(imported.layers[i].values, prefix.state.layers[i].values, **TOL)
        snapshot = adapter.snapshot_state(prefix.state)
        restored = adapter.restore_state(prefix.state, snapshot)
        resumed = adapter.forward(ids[n:], restored)
        assert torch.equal(resumed.output, ordinary.output)
        # A corrupted snapshot (wrong shape family) is refused.
        with pytest.raises(ValueError):
            adapter.restore_state(prefix.state, {"tensors": snapshot["tensors"], "skeleton": {**snapshot["skeleton"], "layer99": {}}})
        with pytest.raises(ValueError):
            adapter.import_self_prefix(prefix.state, {}, torch.arange(n))
        with pytest.raises(ValueError):
            adapter.import_self_prefix(prefix.state, prefix.canonical, torch.arange(1, n + 1))


def test_qwen_and_glm_couple_in_a_live_hive():
    """M2 in-process on the real pair's architectures: two writers, one-epoch lag,
    checkpoint mid-run, restore and reproduce."""
    from uuid import UUID
    from drift.core.pool import PoolBank
    from drift.runtime.hive import HiveController
    from drift.runtime.worker import Worker
    from drift.translate.pool import Layout, Translator, fit_member, fit_pool_format
    from drift.transport.wire2 import FrameCodec2
    session = UUID(int=99)
    _, qwen, _ = make_qwen()
    _, glm, _ = make_glm()
    torch.manual_seed(4)
    calib = torch.randint(3, 97, (40,))
    with torch.no_grad():
        q_out, g_out = qwen.forward(calib), glm.forward(calib)
    qd, gd = qwen.descriptor, glm.descriptor
    q_layout, g_layout = Layout("kv_split", qd.kv_heads, qd.head_dim), Layout("mla_latent", 1, gd.head_dim)
    levels = min(len(qd.kv_layers), len(gd.kv_layers))
    q_rows = {lvl: q_layout.flatten(q_out.canonical[qd.kv_layers[lvl]]) for lvl in range(levels)}
    g_rows = {lvl: g_layout.flatten(g_out.canonical[gd.kv_layers[lvl]]) for lvl in range(levels)}
    fmt, pool_rows = fit_pool_format({lvl: {"qwen": q_rows[lvl], "glm": g_rows[lvl]} for lvl in range(levels)}, width=8)
    q_tr = Translator("qwen", q_layout, {qd.kv_layers[lvl]: lvl for lvl in range(levels)}, fmt)
    g_tr = Translator("glm", g_layout, {gd.kv_layers[lvl]: lvl for lvl in range(levels)}, fmt)
    fit_member(q_tr, q_rows, pool_rows)
    fit_member(g_tr, g_rows, pool_rows)
    workers = {
        "qwen": Worker("qwen", 1, session, qwen, q_tr, PoolBank(session, q_tr, 1, {2}, sinks=2, recent=8), override=0.5),
        "glm": Worker("glm", 2, session, glm, g_tr, PoolBank(session, g_tr, 2, {1}, sinks=2, recent=8), override=0.5),
    }
    controller = HiveController(workers, FrameCodec2(b"q" * 32))
    g = torch.Generator().manual_seed(8)
    inputs = [{n: torch.randint(3, 97, (2,), generator=g) for n in workers} for _ in range(4)]
    with torch.no_grad():
        first = [controller.tick(x) for x in inputs[:2]]
        checkpoint = controller.checkpoint()
        later = [controller.tick(x) for x in inputs[2:]]
        controller.restore(checkpoint)
        again = [controller.tick(x) for x in inputs[2:]]
    assert first[1]["glm"].foreign_tokens == 2 and first[1]["qwen"].foreign_tokens == 2
    assert all(float(m.mean()) > 0 for m in first[1]["glm"].foreign_mass.values())
    for a, b in zip(later, again):
        for name in a:
            assert torch.equal(a[name].output, b[name].output)
    assert workers["glm"].bank.pin().writers.unique().tolist() == [1]


@pytest.mark.parametrize("family", FAMILIES)
def test_embeddings_path_matches_ids_and_marker_writes_private_mail(family):
    from drift.mailbox.stream import MailWriter
    _, adapter, _ = FAMILIES[family]()
    ids = torch.randint(3, 97, (6,))
    with torch.no_grad():
        by_ids = adapter.forward(ids)
        by_embeds = adapter.forward_embeddings(adapter.embed(ids))
        torch.testing.assert_close(by_embeds.output, by_ids.output, **TOL)
        parent = adapter.forward(ids[:3]).state
        before = adapter.snapshot_state(parent)
    writer = MailWriter(adapter.embed(ids[:1]).shape[-1], max_tokens=4)
    canonical = writer.write(adapter, parent, ids[3:5])
    after = adapter.snapshot_state(parent)
    assert all(torch.equal(before["tensors"][k], after["tensors"][k]) for k in before["tensors"])
    rows = next(iter(canonical.values()))
    assert (rows.k.shape[0] if hasattr(rows, "k") else rows.shape[0]) == 3      # marker + 2 tokens
    loss = sum((e.v if hasattr(e, "v") else e).square().sum() for e in canonical.values())
    loss.backward()
    assert writer.marker.grad is not None and writer.marker.grad.abs().sum() > 0


def test_qwen_mails_glm_and_replay_ablates_it():
    from uuid import UUID
    from drift.core.pool import PoolBank
    from drift.eval.replay import ReplayCondition, replay
    from drift.mailbox.stream import MailStatus, MailWriter, MailboxController
    from drift.runtime.hive import HiveController
    from drift.runtime.worker import Worker
    from drift.translate.pool import Layout, Translator, fit_member, fit_pool_format
    from drift.transport.wire2 import FrameCodec2
    session = UUID(int=123)
    _, qwen, _ = make_qwen()
    _, glm, _ = make_glm()
    torch.manual_seed(5)
    calib = torch.randint(3, 97, (40,))
    with torch.no_grad():
        q_out, g_out = qwen.forward(calib), glm.forward(calib)
    qd, gd = qwen.descriptor, glm.descriptor
    q_layout, g_layout = Layout("kv_split", qd.kv_heads, qd.head_dim), Layout("mla_latent", 1, gd.head_dim)
    levels = min(len(qd.kv_layers), len(gd.kv_layers))
    q_rows = {l: q_layout.flatten(q_out.canonical[qd.kv_layers[l]]) for l in range(levels)}
    g_rows = {l: g_layout.flatten(g_out.canonical[gd.kv_layers[l]]) for l in range(levels)}
    fmt, pool_rows = fit_pool_format({l: {"qwen": q_rows[l], "glm": g_rows[l]} for l in range(levels)}, width=8)
    q_tr = Translator("qwen", q_layout, {qd.kv_layers[l]: l for l in range(levels)}, fmt)
    g_tr = Translator("glm", g_layout, {gd.kv_layers[l]: l for l in range(levels)}, fmt)
    fit_member(q_tr, q_rows, pool_rows)
    fit_member(g_tr, g_rows, pool_rows)
    hidden = qwen.embed(calib[:1]).shape[-1]
    workers = {
        "qwen": Worker("qwen", 1, session, qwen, q_tr, PoolBank(session, q_tr, 1, {2}, sinks=2, recent=8), override=0.5,
                       mail_writer=11, mail_bank=PoolBank(session, q_tr, 11, {12}, sinks=0, recent=8, multiple_per_epoch=True), mail=MailWriter(hidden, 4)),
        "glm": Worker("glm", 2, session, glm, g_tr, PoolBank(session, g_tr, 2, {1}, sinks=2, recent=8), override=0.5,
                      mail_writer=12, mail_bank=PoolBank(session, g_tr, 12, {11}, sinks=0, recent=8, multiple_per_epoch=True), mail=MailWriter(glm.embed(calib[:1]).shape[-1], 4)),
    }
    controller, box = HiveController(workers, FrameCodec2(b"z" * 32)), MailboxController(ttl=6, slots=4, per_epoch_cap=2)
    g = torch.Generator().manual_seed(9)
    step_inputs = [{n: torch.randint(3, 97, (2,), generator=g) for n in workers} for _ in range(5)]
    with torch.no_grad():
        controller.tick(step_inputs[0])
        controller.tick(step_inputs[1])
        record = box.create("qwen", epoch=1, tokens=2)
        controller.post_mail("qwen", torch.tensor([40, 41]), box, record.id)
        checkpoint = controller.checkpoint()
        report = controller.tick(step_inputs[2])["glm"]
        mail_mass = {layer: m[report.native_foreign_tokens:] for layer, m in report.entry_mass.items()}
        box.observe("glm", 2, workers["glm"].mail_bank.pin(), mail_mass)
    assert report.mail_foreign_tokens == 3 and box.records[0].status in {MailStatus.VISIBLE, MailStatus.ATTENDED}
    positions = frozenset(workers["glm"].mail_bank.pin().positions.tolist())
    results = replay(controller, checkpoint, "glm", lambda n: workers[n].mail_bank,
                     [ReplayCondition("active"), ReplayCondition("ablated", positions)], step_inputs[2:4])
    assert not torch.allclose(results["active"][-1].output, results["ablated"][-1].output)
    assert results["ablated"][0].mail_foreign_tokens == 0


def test_service_members_build_from_manifest_registry_and_saved_translators(tmp_path):
    """Real-member path end to end on tiny weights saved to disk: checkpoints -> from_local,
    registry entries with real hashes, saved translators -> from_manifest -> a live hive."""
    import hashlib, json
    from uuid import UUID
    from drift.registry import ModelEntry, write_entry
    from drift.runtime.builders import from_manifest
    from drift.runtime.hive import HiveController
    from drift.translate.pool import Layout, Translator, fit_member, fit_pool_format, save_translator
    from drift.transport.wire2 import FrameCodec2
    stock_q, qwen, _ = make_qwen()
    stock_g, glm, _ = make_glm()
    q_dir, g_dir = tmp_path / "qwen", tmp_path / "glm"
    stock_q.save_pretrained(q_dir)
    stock_g.save_pretrained(g_dir)
    # Fit translators on the in-memory adapters, save them.
    torch.manual_seed(2)
    calib = torch.randint(3, 97, (40,))
    with torch.no_grad():
        q_out, g_out = qwen.forward(calib), glm.forward(calib)
    qd, gd = qwen.descriptor, glm.descriptor
    q_layout, g_layout = Layout("kv_split", qd.kv_heads, qd.head_dim), Layout("mla_latent", 1, gd.head_dim)
    levels = min(len(qd.kv_layers), len(gd.kv_layers))
    q_rows = {l: q_layout.flatten(q_out.canonical[qd.kv_layers[l]]) for l in range(levels)}
    g_rows = {l: g_layout.flatten(g_out.canonical[gd.kv_layers[l]]) for l in range(levels)}
    fmt, pool_rows = fit_pool_format({l: {"qwen": q_rows[l], "glm": g_rows[l]} for l in range(levels)}, width=8)
    q_tr = Translator("qwen", q_layout, {qd.kv_layers[l]: l for l in range(levels)}, fmt)
    g_tr = Translator("glm", g_layout, {gd.kv_layers[l]: l for l in range(levels)}, fmt)
    fit_member(q_tr, q_rows, pool_rows)
    fit_member(g_tr, g_rows, pool_rows)
    save_translator(q_tr, tmp_path / "tr_qwen", {"test": True})
    save_translator(g_tr, tmp_path / "tr_glm", {"test": True})
    registry = tmp_path / "registry"
    for model_id, adapter_id, d in (("qwen-tiny", "qwen4_exp/torch", q_dir), ("glm-tiny", "glm5_next/torch", g_dir)):
        cfg = hashlib.sha256((d / "config.json").read_bytes()).hexdigest()
        weights = hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
        write_entry(registry, ModelEntry(model_id, adapter_id, None, "local-tiny", cfg, weights, None, "ab" * 32, "none", "macbook", 2, "cd" * 32))
    manifest = {"kind": "real", "stage": "M0", "registry_root": str(registry), "session_uuid": str(UUID(int=5)),
                "pool": {"levels": fmt.levels, "width": fmt.width, "fingerprint": fmt.fingerprint},
                "window": {"sinks": 2, "recent": 8}, "mail_max_tokens": 4,
                "members": [{"name": "qwen", "writer": 1, "mail_writer": 11, "registry_entry": "qwen-tiny", "checkpoint_dir": str(q_dir), "translator_dir": str(tmp_path / "tr_qwen")},
                            {"name": "glm", "writer": 2, "mail_writer": 12, "registry_entry": "glm-tiny", "checkpoint_dir": str(g_dir), "translator_dir": str(tmp_path / "tr_glm")}]}
    workers = from_manifest(manifest)
    assert set(workers) == {"qwen", "glm"}
    # The reloaded checkpoints reproduce the in-memory adapters exactly.
    ids = torch.randint(3, 97, (5,))
    with torch.no_grad():
        torch.testing.assert_close(workers["qwen"].adapter.forward(ids).output, qwen.forward(ids).output, **TOL)
        torch.testing.assert_close(workers["glm"].adapter.forward(ids).output, glm.forward(ids).output, **TOL)
    controller = HiveController(workers, FrameCodec2(b"r" * 32))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _ in range(2):
            reports = controller.tick({n: torch.randint(3, 97, (2,), generator=g) for n in workers})
    assert reports["glm"].foreign_tokens == 2
    # An entry below the stage's level is refused.
    manifest["stage"] = "M4"
    with pytest.raises(RuntimeError, match="BLOCKED"):
        from_manifest(manifest)


def test_serving_tap_of_a_native_rotated_cache_matches_adapter_canonical():
    """Qualification ladder step 2, emulated: the HF cache holds post-RoPE keys exactly like a
    serving stack's paged cache. Scatter it into pages, tap through the connector core with a
    RopeSpec built from the model config, and compare with the adapter's canonical capture."""
    from drift.serving.core import LayerSpec, RopeSpec, inject_layer, tap_layer
    _, adapter, _ = make_qwen()
    text = adapter._text.config
    params = text.rope_parameters
    rope = RopeSpec(float(params["rope_theta"]), int(text.head_dim * params.get("partial_rotary_factor", 1.0)))
    ids = torch.randint(3, 97, (13,))
    with torch.no_grad():
        out = adapter.forward(ids)
    block = 4
    slots = torch.randperm(8 * block)[:13]                    # scattered block table
    for layer in adapter.descriptor.kv_layers:
        cache_layer = out.state.layers[layer]
        native_k = cache_layer.keys[0].transpose(0, 1)        # [T, H, D], rotated
        native_v = cache_layer.values[0].transpose(0, 1)
        paged = torch.zeros(8, 2, block, native_k.shape[1], native_k.shape[2])
        paged[slots // block, 0, slots % block] = native_k
        paged[slots // block, 1, slots % block] = native_v
        spec = LayerSpec(f"layers.{layer}", layer, "blocks_first", rope)
        tapped = tap_layer(spec, paged, slots, torch.arange(13))
        torch.testing.assert_close(tapped.k, out.canonical[layer].k, **TOL)
        torch.testing.assert_close(tapped.v, out.canonical[layer].v, **TOL)
        # and the reverse: injecting canonical entries reproduces the native rotated rows
        fresh = torch.zeros_like(paged)
        inject_layer(spec, fresh, slots, torch.arange(13), out.canonical[layer])
        torch.testing.assert_close(fresh[slots // block, 0, slots % block], native_k, **TOL)
