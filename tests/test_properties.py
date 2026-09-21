"""Property and fuzz coverage added after an adversarial review of the kit.

These go beyond the kit's spot checks: attention is compared with an independent
dense reference over random shapes and masks, the bank window is compared with a
brute-force model, the wire codec is fuzzed with corrupted and re-signed frames,
and decoder parity is checked at several lengths. Seeded, so deterministic.
"""
from __future__ import annotations
import hmac
import math
import random
import struct
from types import MappingProxyType
from uuid import UUID
import pytest
import torch
from drift.core.attention import Gate, attend
from drift.core.memory import ForeignKVBank, ForeignView
from drift.core.position import recency_positions
from drift.core.projector import BridgeProjector
from drift.core.types import Delta, KV
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.transport.wire import FrameCodec

SESSION = UUID(int=41)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    random.seed(0)


def identity(heads: int, dim: int) -> BridgeProjector:
    projector = BridgeProjector((heads, dim), (heads, dim))
    with torch.no_grad():
        for sub in (projector.k_map, projector.v_map):
            sub.map.weight.copy_(torch.eye(heads * dim))
            sub.map.bias.zero_()
    return projector


def dense_reference(q, kn, vn, mn, kf, vf, mf, gate_value):
    """Per-query, per-head loop: expand GQA, concatenate, shared softmax with log g."""
    queries, heads, dim = q.shape
    repeat = heads // kn.shape[1]
    kn, vn = kn.repeat_interleave(repeat, 1), vn.repeat_interleave(repeat, 1)
    kf, vf = kf.repeat_interleave(repeat, 1), vf.repeat_interleave(repeat, 1)
    out = torch.zeros(queries, heads, dim, dtype=torch.float64)
    mass = torch.zeros(queries, heads, dtype=torch.float64)
    minus_inf = torch.tensor(-math.inf, dtype=torch.float64)
    for i in range(queries):
        for h in range(heads):
            sn = torch.where(mn[i], (kn[:, h] @ q[i, h]) / math.sqrt(dim), minus_inf)
            sf = torch.where(mf[i], (kf[:, h] @ q[i, h]) / math.sqrt(dim) + math.log(gate_value), minus_inf)
            w = torch.cat((sn, sf)).softmax(0)
            out[i, h] = w[: len(sn)] @ vn[:, h] + w[len(sn):] @ vf[:, h]
            mass[i, h] = w[len(sn):].sum()
    return out, mass


def test_attention_matches_dense_reference_over_random_shapes_masks_and_gates():
    for _ in range(60):
        queries = random.randint(1, 4)
        kv_heads, repeat, dim = random.choice([1, 2]), random.choice([1, 2, 3]), random.choice([4, 8])
        tn, tf = random.randint(queries, 6), random.randint(1, 5)
        q = torch.randn(queries, kv_heads * repeat, dim, dtype=torch.float64)
        native = KV(torch.randn(tn, kv_heads, dim, dtype=torch.float64), torch.randn(tn, kv_heads, dim, dtype=torch.float64))
        foreign = KV(torch.randn(tf, kv_heads, dim, dtype=torch.float64), torch.randn(tf, kv_heads, dim, dtype=torch.float64))
        mn = torch.rand(queries, tn) < 0.7
        mn[:, 0] = True
        mf = torch.rand(queries, tf) < 0.7
        g = random.choice([1.0, 0.5, 0.01])
        out, mass = attend(q, native, mn, foreign, override=g, allowed_foreign=mf)
        ref, ref_mass = dense_reference(q, native.k, native.v, mn, foreign.k, foreign.v, mf, g)
        if not mf.any():
            ref_mass = torch.zeros_like(ref_mass)
        # attend() forms scores in float32 even for float64 inputs, so float32 tolerance applies.
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(mass, ref_mass, rtol=1e-5, atol=1e-6)
        gate = Gate(math.log(g / (1 - g)) if g < 1 else 30.0)
        gated, _ = attend(q, native, mn, foreign, gate=gate, allowed_foreign=mf)
        torch.testing.assert_close(gated, out, rtol=1e-6, atol=1e-6)


def test_bank_window_equals_brute_force_sinks_plus_recent_model():
    heads, dim = 2, 4
    for _ in range(40):
        sinks, recent = random.randint(0, 3), random.randint(1, 6)
        bank = ForeignKVBank(SESSION, 0, [(0, 0, identity(heads, dim)), (3, 1, identity(heads, dim))],
                             sinks=sinks, recent=recent)
        history: dict[int, tuple[int, dict[int, KV]]] = {}
        start = 0
        for epoch in range(random.randint(1, 8)):
            tokens = random.randint(1, 5)
            layers = {0: KV(torch.randn(tokens, heads, dim), torch.randn(tokens, heads, dim)),
                      3: KV(torch.randn(tokens, heads, dim), torch.randn(tokens, heads, dim))}
            for j in range(tokens):
                history[start + j] = (j, layers)
            bank.commit(Delta(SESSION, 0, epoch, epoch, start, {k: v.clone() for k, v in layers.items()}))
            start += tokens
            latest = start - 1
            expected = sorted(p for p in history if p < sinks or p >= latest - recent + 1)
            view = bank.pin()
            assert view is not None
            assert view.positions.tolist() == expected
            for i, position in enumerate(expected):
                offset, source = history[position]
                torch.testing.assert_close(view.layers[0].k[i], source[0].k[offset])
                torch.testing.assert_close(view.layers[1].v[i], source[3].v[offset])
            virtual = recency_positions(view.positions, 100)
            assert virtual[-1] == 99 and bool(torch.all(virtual[1:] > virtual[:-1]))


def test_corrupted_and_resigned_frames_never_decode_and_never_crash():
    key = b"k" * 32
    codec = FrameCodec(key)
    # Header layout: magic[8] version[1] direction[1] session[16] epoch[8] seq[8] start[8] tokens[4] count[2]
    fields = [("tokens", 50, "!I"), ("count", 54, "!H"), ("layer_heads", 58, "!H"),
              ("layer_dim", 60, "!H"), ("magic", 0, None), ("version", 8, "!B")]
    for trial in range(300):
        tokens, count = random.randint(1, 4), random.randint(1, 3)
        layers = {i: KV(torch.randn(tokens, 2, 4), torch.randn(tokens, 2, 4)) for i in range(count)}
        frame = bytearray(codec.encode(Delta(SESSION, 0, trial, trial, 0, layers)))
        mode = random.choice(["flip", "truncate", "extend", "resign"])
        if mode == "flip":
            frame[random.randrange(len(frame))] ^= 1 << random.randrange(8)
        elif mode == "truncate":
            frame = frame[: random.randrange(len(frame))]
        elif mode == "extend":
            frame += bytes(random.randrange(1, 40))
        else:
            body = bytearray(frame[:-32])
            _, offset, fmt = random.choice(fields)
            if fmt is None:
                body[0:8] = b"NOTMAGIC"
            else:
                (original,) = struct.unpack_from(fmt, body, offset)
                limit = 0xFF if fmt == "!B" else 0xFFFF
                candidates = [v for v in (0, 1, 2, 3, 0xFF, limit, (1 << 31) if fmt == "!I" else 7) if v != original]
                struct.pack_into(fmt, body, offset, random.choice(candidates))
            frame = bytes(body) + hmac.digest(key, bytes(body), "sha256")
        with pytest.raises(ValueError):
            codec.decode(bytes(frame))


@pytest.mark.parametrize("n", [1, 2, 7, 16, 31])
def test_decoder_parity_multi_token_identity_and_hard_off(n):
    torch.set_num_threads(1)
    worker = FrozenDecoder(ToyModel())
    vocab = worker.model.config.vocab_size
    ids = torch.randint(0, vocab, (n + 5,))
    with torch.no_grad():
        full = worker.forward(ids)
        prefix = worker.forward(ids[:n])
        continued = worker.forward(ids[n:], prefix.state)
        torch.testing.assert_close(full.logits[n:], continued.logits, rtol=1e-5, atol=2e-5)
        state, steps = None, []
        for token in ids:
            output = worker.forward(token[None], state)
            state, steps = output.state, steps + [output.logits]
        torch.testing.assert_close(full.logits, torch.cat(steps), rtol=1e-5, atol=2e-5)
        imported = worker.import_self_prefix(prefix.canonical_delta, torch.arange(n))
        handoff = worker.forward(ids[n:], imported)
        assert torch.equal(handoff.logits, continued.logits)
        layers, heads, dim = len(worker.layers), worker.kvheads, worker.dim
        hostile = ForeignView(0, torch.arange(3), MappingProxyType({
            i: KV(torch.full((3, heads, dim), 1e3), torch.full((3, heads, dim), -1e3)) for i in range(layers)}))
        closed = worker.forward(ids[n:], prefix.state, foreign=hostile, override=0.0)
        assert torch.equal(closed.logits, continued.logits)
        for ours, theirs in zip(closed.state.layers, continued.state.layers):
            assert torch.equal(ours.k, theirs.k) and torch.equal(ours.v, theirs.v)
        opened = worker.forward(ids[n:], prefix.state, foreign=hostile, override=1.0)
        assert not torch.allclose(opened.logits, continued.logits)


def test_recency_positions_preserve_source_distances():
    for _ in range(50):
        positions = torch.sort(torch.randperm(200)[: random.randint(1, 20)]).values
        anchor = random.randint(0, 50)
        virtual = recency_positions(positions, anchor)
        assert virtual[-1] == anchor - 1
        assert torch.equal(virtual[1:] - virtual[:-1], positions[1:] - positions[:-1])
