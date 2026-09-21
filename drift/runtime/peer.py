"""Cross-process lockstep over real sockets (spec M2.2 remote, M4.1 TCP backend).

Each process hosts one Worker. Per epoch both peers: pin -> compute -> send their
publication -> receive the peer's -> validate and commit -> advance. Publications go
through `FrameCodec2` with a u32 length prefix (wire v2 framing). A peer never reads
a same-epoch publication: it commits the peer's epoch-k block only after its own
epoch-k compute finished, and the bank rejects anything out of order. Any failure
poisons the local worker; there is no silent resume.

Use loopback or an approved tunnel; HMAC authenticates, it does not encrypt.
"""
from __future__ import annotations
import socket
import struct
from dataclasses import dataclass
import torch
from drift.runtime.worker import StepReport, Worker
from drift.transport.wire2 import FrameCodec2, Publication


@dataclass
class PeerLink:
    sock: socket.socket
    codec: FrameCodec2
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.sock.settimeout(self.timeout)
        self.closed = False

    def _read(self, length: int) -> bytes:
        chunks, remaining = [], length
        while remaining:
            chunk = self.sock.recv(min(remaining, 65536))
            if not chunk:
                raise EOFError("peer disconnected before a complete frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send(self, pub: Publication) -> int:
        frame = self.codec.encode(pub)
        self.sock.sendall(struct.pack("!I", len(frame)) + frame)
        return len(frame) + 4

    def receive(self) -> tuple[Publication, int]:
        (size,) = struct.unpack("!I", self._read(4))
        if not 0 < size <= self.codec.max_frame:
            raise ValueError("announced frame length exceeds limit")
        frame = self._read(size)
        return self.codec.decode(frame), size + 4

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.sock.close()


class PeerRunner:
    """One side of a two-process session."""
    def __init__(self, worker: Worker, link: PeerLink):
        self.worker, self.link = worker, link
        self.epoch = 0
        self.failed = False
        self.wire_bytes_sent = 0
        self.wire_bytes_received = 0

    def tick(self, local_ids: torch.Tensor) -> StepReport:
        if self.failed:
            raise RuntimeError("peer runner is poisoned")
        try:
            view = self.worker.bank.pin()
            if view is not None and view.epoch != self.epoch - 1:
                raise ValueError("lockstep requires precisely the prior epoch")
            report = self.worker.step(local_ids, self.epoch)
            self.wire_bytes_sent += self.link.send(report.publication)
            received, nbytes = self.link.receive()
            self.wire_bytes_received += nbytes
            if received.epoch != self.epoch:
                raise ValueError("peer publication is not from this epoch")
            self.worker.bank.commit(received)
            self.epoch += 1
            return report
        except Exception:
            self.failed = True
            self.worker.poisoned = True
            self.link.close()
            raise


def connect_pair(port: int = 0, timeout: float = 30.0) -> tuple[socket.socket, socket.socket]:
    """Loopback helper for tests: returns (server-side, client-side) connected sockets."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(timeout)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=timeout)
    server, _ = listener.accept()
    listener.close()
    return server, client
