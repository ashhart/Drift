"""Wire v2 and the multi-writer pool bank, on synthetic translators."""
from __future__ import annotations
import hmac
import random
import struct
from dataclasses import replace
from uuid import UUID
import pytest
import torch
from drift.core.pool import PoolBank
from drift.core.types import KV
from drift.translate.pool import Layout, PoolFormat, Translator
from drift.transport.wire2 import HEADER2, MAGIC2, FrameCodec2, Publication

SESSION = UUID(int=41)
FMT = PoolFormat("pool.v1", 2, 6, "ab" * 32)


def reader(kind="kv_split") -> Translator:
    layout = Layout("kv_split", 2, 3) if kind == "kv_split" else Layout("mla_latent", 1, 5)
    t = Translator("R", layout, {1: 0, 3: 1}, FMT)
    with torch.no_grad():
        for level in ("0", "1"):
            t.readers[level].linear.weight.copy_(torch.randn(layout.width, 6))
            t.readers[level].linear.bias.zero_()
    return t


def pub(writer=1, epoch=0, seq=0, start=0, tokens=2, fp=FMT.fingerprint):
    return Publication(SESSION, writer, fp, epoch, seq, start,
                       {0: torch.randn(tokens, 6), 1: torch.randn(tokens, 6)})


def test_codec_roundtrip_and_structure():
    codec = FrameCodec2(b"k" * 32)
    original = pub()
    frame = codec.encode(original)
    assert frame.startswith(MAGIC2) and len(frame) == HEADER2.size + 2 * (6 + 2 * 6 * 4) + 32
    back = codec.decode(frame)
    assert (back.session, back.writer, back.fingerprint, back.epoch, back.sequence, back.start) == \
           (SESSION, 1, FMT.fingerprint, 0, 0, 0)
    torch.testing.assert_close(back.levels[1], original.levels[1])
    with pytest.raises(ValueError):
        FrameCodec2(b"j" * 32).decode(frame)


def test_codec_fuzz_never_decodes_corruption():
    random.seed(1)
    key = b"k" * 32
    codec = FrameCodec2(key)
    # header: magic8 ver1 writer2 session16 fp32 epoch8 seq8 start8 tokens4 levels2 = 89 bytes -> tokens@83, levels@87, first level width@91
    fields = [("tokens", 83, "!I"), ("levels", 87, "!H"), ("level_width", 91, "!I"), ("magic", 0, None), ("version", 8, "!B")]
    for trial in range(300):
        frame = bytearray(codec.encode(pub(epoch=trial, seq=trial, start=2 * trial, tokens=random.randint(1, 4))))
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
                candidates = [v for v in (0, 1, 2, 3, limit, (1 << 31) if fmt == "!I" else 7) if v != original]
                struct.pack_into(fmt, body, offset, random.choice(candidates))
            frame = bytes(body) + hmac.digest(key, bytes(body), "sha256")
        with pytest.raises(ValueError):
            codec.decode(bytes(frame))


def test_bank_appends_from_two_writers_excludes_self_and_bounds_window():
    torch.manual_seed(0)
    bank = PoolBank(SESSION, reader(), self_writer=3, writers={1, 2}, sinks=1, recent=3)
    assert bank.pin() is None
    bank.commit(pub(writer=1, epoch=0, seq=0, start=0, tokens=2))
    bank.commit(pub(writer=2, epoch=0, seq=0, start=0, tokens=2))
    old = bank.pin()
    assert old.positions.tolist() == [0, 1, 2, 3] and old.writers.tolist() == [1, 1, 2, 2]
    assert isinstance(old.layers[1], KV) and old.layers[1].k.shape == (4, 2, 3)
    bank.commit(pub(writer=1, epoch=1, seq=1, start=2, tokens=3))
    view = bank.pin()
    assert view.positions.tolist() == [0, 4, 5, 6] and view.writers.tolist() == [1, 1, 1, 1]
    assert old.positions.tolist() == [0, 1, 2, 3]            # pinned view untouched
    with pytest.raises(ValueError, match="self publication"):
        bank.commit(pub(writer=3))
    with pytest.raises(ValueError, match="unknown writer"):
        bank.commit(pub(writer=9))


@pytest.mark.parametrize("mutation", ["session", "fingerprint", "sequence", "start", "epoch", "levels"])
def test_bad_publication_leaves_bank_unchanged(mutation):
    bank = PoolBank(SESSION, reader("mla_latent"), self_writer=None, writers={1})
    bank.commit(pub(writer=1, epoch=0, seq=0, start=0, tokens=2))
    good = pub(writer=1, epoch=1, seq=1, start=2, tokens=1)
    values = {"session": UUID(int=9), "fingerprint": "cd" * 32, "sequence": 0, "start": 3, "epoch": 0,
              "levels": {0: torch.randn(1, 6)}}
    old = bank.pin()
    with pytest.raises(ValueError):
        bank.commit(replace(good, **{mutation: values[mutation]}))
    assert bank.pin() is old and bank.cursors[1].next_sequence == 1 and bank.next_slot == 2
    assert old.layers[1].shape == (2, 5)


def test_bank_snapshot_restore_is_lossless():
    torch.manual_seed(2)
    bank = PoolBank(SESSION, reader(), self_writer=None, writers={1, 2}, sinks=2, recent=4)
    bank.commit(pub(writer=1, epoch=0, seq=0, start=0, tokens=3))
    bank.commit(pub(writer=2, epoch=0, seq=0, start=0, tokens=2))
    snap = bank.snapshot()
    nxt = pub(writer=1, epoch=1, seq=1, start=3, tokens=2)
    bank.commit(nxt.clone())
    after = bank.pin()
    other = PoolBank(SESSION, bank.reader, self_writer=None, writers={1, 2}, sinks=2, recent=4)
    other.restore(snap)
    other.commit(nxt.clone())
    assert torch.equal(other.pin().positions, after.positions)
    for layer in after.layers:
        assert torch.equal(other.pin().layers[layer].k, after.layers[layer].k)
    with pytest.raises(ValueError):
        PoolBank(SESSION, bank.reader, None, writers={1}).restore(snap)
