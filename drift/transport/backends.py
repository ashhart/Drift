from __future__ import annotations
import queue
import socket
import struct
from typing import Protocol
from drift.core.types import Delta
from .wire import FrameCodec


class Transport(Protocol):
    def send(self, delta: Delta) -> None: ...
    def receive(self) -> Delta: ...
    def close(self) -> None: ...


class InProcTransport:
    """Bounded COPY baseline using exactly the network wire codec."""
    def __init__(self, codec: FrameCodec, capacity: int = 2, timeout: float = 2.0):
        if capacity <= 0 or timeout <= 0:
            raise ValueError("positive capacity and timeout required")
        self.codec, self.timeout = codec, timeout
        self.queue: queue.Queue[bytes] = queue.Queue(capacity)
        self.closed = False

    def send(self, delta: Delta) -> None:
        if self.closed:
            raise RuntimeError("transport closed")
        self.queue.put(self.codec.encode(delta), timeout=self.timeout)

    def receive(self) -> Delta:
        if self.closed:
            raise RuntimeError("transport closed")
        return self.codec.decode(self.queue.get(timeout=self.timeout))

    def close(self) -> None:
        self.closed = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class SocketTransport:
    """Length-framed TCP/socket reference; closes on partial-frame failures.

    Only use raw sockets on loopback or within an approved protected tunnel.
    For nonlocal production traffic wrap a connected socket in mutual TLS, then
    pass it here. HMAC alone is not confidentiality.
    """
    def __init__(self, sock: socket.socket, codec: FrameCodec, timeout: float = 2.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.sock, self.codec = sock, codec
        self.sock.settimeout(timeout)
        self.closed = False

    def _read(self, length: int) -> bytes:
        chunks, remaining = [], length
        while remaining:
            chunk = self.sock.recv(min(remaining, 65536))
            if not chunk:
                raise EOFError("peer disconnected before complete frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send(self, delta: Delta) -> None:
        if self.closed:
            raise RuntimeError("transport closed")
        frame = self.codec.encode(delta)
        try:
            self.sock.sendall(struct.pack("!I", len(frame)) + frame)
        except (OSError, TimeoutError):
            self.close()
            raise

    def receive(self) -> Delta:
        if self.closed:
            raise RuntimeError("transport closed")
        try:
            size, = struct.unpack("!I", self._read(4))
            if not 0 < size <= self.codec.max_frame:
                raise ValueError("announced frame length exceeds limit")
            return self.codec.decode(self._read(size))
        except (OSError, EOFError, ValueError):
            self.close()
            raise

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.sock.close()


class MCDMATransport:
    """Deliberately unavailable, not a silent TCP fallback or invented vendor API.

    M4 replaces this guard after inspecting the user's pinned MCDMA source,
    native buffer registration API, completion semantics and GPU visibility.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "M4 BLOCKED: native MCDMA integration is not supplied or hardware-validated. "
            "Implement the registration/completion contract in the engineering file. "
            "Select inproc or an explicitly declared TCP run instead.")
