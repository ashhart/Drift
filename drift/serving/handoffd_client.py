"""Drift over MCDMA: pull a writer's cache export from a Spark into Studio memory through the owner's `handoffd`.

`handoffd` (owner's RDMA daemon, one source for both sides) already does the transport: the Spark side registers the export
and serves it, the Studio side RDMA-READs it over the MCDMA provider into a POSIX shared-memory object `/q38_<peer>` and
answers on a Unix socket. This client only speaks that socket and maps the memory READ-ONLY. It opens NO verbs objects
itself, so it can exit or crash without touching a queue pair (a Studio process dying with a live QP once panicked the Mac;
only `handoffd` may own verbs, and it is stopped with SHUTDOWN on its socket, never killed)."""
from __future__ import annotations
import mmap
import os
import re
import socket
from dataclasses import dataclass

_PEER, _PATH = re.compile(r"[A-Za-z0-9_.-]{1,32}"), re.compile(r"/[A-Za-z0-9_./-]{1,480}")


@dataclass(frozen=True)
class Pull:
    peer: str
    bytes: int
    loop_ns: int          # RDMA READs + copies
    copy_ns: int
    gbit_s: float
    serve_ns: int
    job_ns: int           # control round trips, destination setup and release too


class HandoffdError(RuntimeError):
    pass


class HandoffdClient:
    def __init__(self, socket_path: str = "/tmp/handoffd.sock", timeout_s: float = 60.0):
        self.socket_path, self.timeout_s = socket_path, timeout_s

    def _exchange(self, line: str) -> list[str]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout_s)
            s.connect(self.socket_path)
            s.sendall(line.encode() + b"\n")
            data = b""
            while not data.endswith(b"END\n") and len(data) < 1 << 20:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        lines = data.decode(errors="replace").splitlines()
        if not lines or lines[-1] != "END":
            raise HandoffdError("handoffd closed the connection before END")
        return lines[:-1]

    def status(self) -> list[str]:
        return self._exchange("STATUS")

    def pull_file(self, peer: str, remote_path: str, local_path: str, unlink: bool = True) -> Pull:
        """RDMA-READ one remote file into a local file, for exports larger than the peer's shared memory."""
        if not _PEER.fullmatch(peer) or not _PATH.fullmatch(remote_path) or ".." in remote_path or not _PATH.fullmatch(local_path) or ".." in local_path:
            raise ValueError("invalid peer or path")
        for line in self._exchange(f"PULL {peer} {remote_path} {local_path} {int(unlink)}"):
            parts = line.split()
            if parts[:2] == ["OK", peer] and len(parts) >= 8:
                return Pull(peer, int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5]), int(parts[6]), int(parts[7]))
            if parts and parts[0] == "ERR":
                raise HandoffdError(" ".join(parts[1:])[:200])
        raise HandoffdError("no result line for the pull")

    def test_orphan(self, peer: str) -> None:
        """Recovery tests only: the daemon forgets the peer's control connection without closing it, as one that died
        silently would be left; the next pull must reconnect on its own."""
        if not _PEER.fullmatch(peer):
            raise ValueError("invalid peer")
        lines = self._exchange(f"TEST_ORPHAN {peer}")
        if lines != [f"ORPHANED {peer}"]:
            raise HandoffdError(" ".join(lines)[:200] or "no reply")

    def pull(self, peer: str, remote_path: str, offset: int = 0, unlink: bool = True) -> Pull:
        """RDMA-READ one remote file into the peer's shared memory at `offset` (4096-aligned)."""
        if not _PEER.fullmatch(peer) or not _PATH.fullmatch(remote_path) or ".." in remote_path or offset < 0 or offset % 4096:
            raise ValueError("invalid peer, remote path or offset")
        for line in self._exchange(f"PULL {peer} {remote_path} shm:{offset} {int(unlink)}"):
            parts = line.split()
            if parts[:2] == ["OK", peer] and len(parts) >= 8:
                return Pull(peer, int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5]), int(parts[6]), int(parts[7]))
            if parts and parts[0] == "ERR":
                raise HandoffdError(" ".join(parts[1:])[:200])
        raise HandoffdError("no result line for the pull")

    @staticmethod
    def view(peer: str, offset: int, length: int) -> memoryview:
        """Read-only view of what a pull wrote. The caller copies out what it keeps; the mapping dies with the view."""
        import _posixshmem
        if not _PEER.fullmatch(peer):
            raise ValueError("invalid peer")
        fd = _posixshmem.shm_open(f"/q38_{peer}", os.O_RDONLY, 0o600)
        try:
            size = os.fstat(fd).st_size
            if offset < 0 or length <= 0 or offset + length > size:
                raise ValueError("requested range is outside the shared memory object")
            mapped = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        return memoryview(mapped)[offset:offset + length]
