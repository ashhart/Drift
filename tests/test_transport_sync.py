import hmac
import queue
import socket
import struct
from uuid import UUID
import pytest
import torch
from drift.core.types import Delta, KV
from drift.core.memory import ForeignKVBank
from drift.core.sync import SyncController
from drift.transport.wire import FrameCodec, HEADER, MAGIC
from drift.transport.backends import InProcTransport, SocketTransport, MCDMATransport
from test_core import identity

SESSION = UUID(int=41)


def delta(epoch=0, direction=0):
    return Delta(SESSION, direction, epoch, epoch, epoch,
                 {0: KV(torch.randn(1, 2, 4), torch.randn(1, 2, 4))})


def test_binary_roundtrip_and_no_plaintext_metadata():
    codec, original = FrameCodec(b"x" * 32), delta()
    frame = codec.encode(original); restored = codec.decode(frame)
    assert frame.startswith(MAGIC)
    assert b"token_ids" not in frame and b"question" not in frame
    assert restored.session == original.session
    torch.testing.assert_close(restored.layers[0].k, original.layers[0].k)


def test_authentication_and_wrong_key_rejected():
    codec = FrameCodec(b"x" * 32); encoded = codec.encode(delta())
    with pytest.raises(ValueError): codec.decode(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    with pytest.raises(ValueError): FrameCodec(b"y" * 32).decode(encoded)


def test_signed_trailing_bytes_and_oversized_dimensions_rejected():
    key, codec = b"x" * 32, FrameCodec(b"x" * 32)
    body = codec.encode(delta())[:-32] + b"untyped"
    with pytest.raises(ValueError): codec.decode(body + hmac.digest(key, body, "sha256"))
    body = HEADER.pack(MAGIC, 1, 0, SESSION.bytes, 0, 0, 0, 900000, 1)
    with pytest.raises(ValueError): codec.decode(body + hmac.digest(key, body, "sha256"))


def test_bounded_inproc_queue_and_closed_guard():
    transport = InProcTransport(FrameCodec(b"x" * 32), capacity=1, timeout=0.01)
    original = delta(); transport.send(original)
    original.layers[0].k.fill_(333)
    with pytest.raises(queue.Full): transport.send(delta(1))
    assert not torch.equal(transport.receive().layers[0].k, original.layers[0].k)
    transport.close()
    with pytest.raises(RuntimeError): transport.send(delta())


def test_socketpair_roundtrip():
    a, b = socket.socketpair()
    tx, rx = SocketTransport(a, FrameCodec(b"x" * 32)), SocketTransport(b, FrameCodec(b"x" * 32))
    try:
        original = delta(); tx.send(original); received = rx.receive()
        torch.testing.assert_close(received.layers[0].v, original.layers[0].v)
    finally:
        tx.close(); rx.close()


def test_actual_loopback_tcp_roundtrip():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(2)
    try:
        listener.bind(("127.0.0.1", 0)); listener.listen(1)
        client = socket.create_connection(listener.getsockname(), timeout=2)
        server, _ = listener.accept()
    except PermissionError:
        listener.close(); pytest.skip("sandbox forbids loopback TCP bind")
    tx, rx = SocketTransport(client, FrameCodec(b"x" * 32)), SocketTransport(server, FrameCodec(b"x" * 32))
    try:
        original = delta(); tx.send(original)
        torch.testing.assert_close(rx.receive().layers[0].k, original.layers[0].k)
    finally:
        tx.close(); rx.close(); listener.close()


def test_truncated_frame_poisons_socket():
    a, b = socket.socketpair(); receiver = SocketTransport(b, FrameCodec(b"x" * 32))
    a.sendall(struct.pack("!I", 100) + b"short"); a.close()
    with pytest.raises(EOFError): receiver.receive()
    assert receiver.closed


def test_mcdma_fails_loudly_instead_of_faking_hardware():
    with pytest.raises(RuntimeError, match="M4 BLOCKED"): MCDMATransport()


def test_causal_scheduler_pins_both_views_before_computation():
    a = ForeignKVBank(SESSION, 1, [(0, 0, identity())])
    b = ForeignKVBank(SESSION, 0, [(0, 0, identity())])
    scheduler, seen = SyncController(a, b), []
    def worker(direction):
        def run(view, epoch):
            seen.append((direction, epoch, None if view is None else view.epoch))
            return delta(epoch, direction)
        return run
    for _ in range(3): scheduler.tick(worker(0), worker(1))
    assert seen == [(0, 0, None), (1, 0, None), (0, 1, 0), (1, 1, 0), (0, 2, 1), (1, 2, 1)]


def test_failed_scheduler_cannot_silently_resume():
    a = ForeignKVBank(SESSION, 1, [(0, 0, identity())]); b = ForeignKVBank(SESSION, 0, [(0, 0, identity())])
    scheduler = SyncController(a, b)
    def fail(view, epoch): raise RuntimeError("injected failure")
    with pytest.raises(RuntimeError): scheduler.tick(fail, fail)
    assert scheduler.failed
    with pytest.raises(RuntimeError, match="poisoned"): scheduler.tick(fail, fail)
