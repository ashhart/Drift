"""Session store shared by serving connectors: authenticated wire-v2 frames on disk.

A writer-side connector appends one frame per publication; a reader-side connector (or a
translator process between them) reads them in order. Frames are the same HMAC'd wire v2
bytes the live transports carry, so nothing but pool rows and typed counters is stored.
Files are written atomically (temp + rename) so a reader never sees a partial frame.
"""
from __future__ import annotations
import os
from pathlib import Path
from uuid import UUID
from drift.transport.wire2 import FrameCodec2, Publication


class SessionStore:
    def __init__(self, root: Path, codec: FrameCodec2):
        self.root, self.codec = Path(root), codec
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, session: UUID, writer: int) -> Path:
        path = self.root / str(session) / f"writer-{writer:05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append(self, pub: Publication) -> Path:
        directory = self._dir(pub.session, pub.writer)
        target = directory / f"{pub.sequence:012d}.frame"
        if target.exists():
            raise FileExistsError("publication sequence already stored; duplicates are refused")
        temp = directory / f".{pub.sequence:012d}.{os.getpid()}.tmp"
        temp.write_bytes(self.codec.encode(pub))
        os.replace(temp, target)
        return target

    def read_from(self, session: UUID, writer: int, next_sequence: int) -> list[Publication]:
        """Publications with sequence >= next_sequence, contiguous from it; stops at the first gap."""
        directory = self._dir(session, writer)
        out = []
        while True:
            path = directory / f"{next_sequence + len(out):012d}.frame"
            if not path.exists():
                return out
            pub = self.codec.decode(path.read_bytes())
            if (pub.session, pub.writer, pub.sequence) != (session, writer, next_sequence + len(out)):
                raise ValueError("stored frame does not match its location")
            out.append(pub)
