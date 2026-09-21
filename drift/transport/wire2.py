"""Wire v2: pool-format publications with a writer id and a format fingerprint.

A v2 frame carries one writer's contiguous block of pool rows for every level it
participates in (HIVE_MIND H1/H2). Rows are already in the pool format (writer
applied), so a receiver only validates, reads (its own reader) and appends. The
frame has no text, token IDs, labels or free metadata.

Header, network byte order, 100 bytes:
  magic[8]="TELEKV02", version:u8, writer:u16, session[16], fingerprint[32],
  epoch:u64, sequence:u64, start:u64, token_count:u32, level_count:u16
For each level: level:u16, width:u32, then rows: little-endian float32 [T, width]
Trailer: HMAC-SHA256 (32 bytes). TCP framing adds a big-endian u32 length.
"""
from __future__ import annotations
import hmac
import struct
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID
import numpy as np
import torch

HEADER2 = struct.Struct("!8sBH16s32sQQQIH")
LEVEL2 = struct.Struct("!HI")
MAGIC2 = b"TELEKV02"
MAX_FRAME = 64 * 1024 * 1024
MAX_TOKENS, MAX_LEVELS, MAX_WIDTH = 8192, 128, 65536


@dataclass(frozen=True)
class Publication:
    """One writer's complete pool block for all its levels; contiguous source slots."""
    session: UUID
    writer: int
    fingerprint: str            # pool format fingerprint (64 hex chars)
    epoch: int
    sequence: int
    start: int
    levels: Mapping[int, torch.Tensor]      # level -> [T, width] float rows

    def check(self) -> None:
        if not 0 <= self.writer <= 65535:
            raise ValueError("bad writer id")
        if len(self.fingerprint) != 64 or any(c not in "0123456789abcdef" for c in self.fingerprint):
            raise ValueError("fingerprint must be 64 hex chars")
        if min(self.epoch, self.sequence, self.start) < 0 or not self.levels:
            raise ValueError("negative counter or empty level set")
        widths, tokens = set(), set()
        for level, rows in self.levels.items():
            if type(level) is not int or not 0 <= level <= 65535:
                raise ValueError("bad level index")
            if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or min(rows.shape) <= 0:
                raise ValueError("rows must be nonempty [T, width]")
            if not rows.is_floating_point() or not torch.isfinite(rows).all():
                raise ValueError("rows must be finite floating point")
            tokens.add(rows.shape[0])
            widths.add(rows.shape[1])
        if len(tokens) != 1:
            raise ValueError("a publication must be complete across levels")

    @property
    def tokens(self) -> int:
        return next(iter(self.levels.values())).shape[0]

    def clone(self) -> Publication:
        return Publication(self.session, self.writer, self.fingerprint, self.epoch, self.sequence,
                           self.start, {k: v.detach().clone() for k, v in self.levels.items()})


class FrameCodec2:
    def __init__(self, key: bytes, max_frame: int = MAX_FRAME):
        if len(key) < 32 or not 1024 <= max_frame <= MAX_FRAME:
            raise ValueError("use >=32-byte secret and bounded frame capacity")
        self.key, self.max_frame = key, max_frame

    def encode(self, pub: Publication) -> bytes:
        pub.check()
        if not 1 <= pub.tokens <= MAX_TOKENS or len(pub.levels) > MAX_LEVELS:
            raise ValueError("frame dimension limit exceeded")
        size = HEADER2.size + 32 + sum(LEVEL2.size + rows.numel() * 4 for rows in pub.levels.values())
        if size > self.max_frame:
            raise ValueError("frame too large; chunk the publication")
        body = bytearray(HEADER2.pack(MAGIC2, 2, pub.writer, pub.session.bytes, bytes.fromhex(pub.fingerprint),
                                      pub.epoch, pub.sequence, pub.start, pub.tokens, len(pub.levels)))
        for level, rows in sorted(pub.levels.items()):
            width = rows.shape[1]
            if width > MAX_WIDTH:
                raise ValueError("row width exceeds wire limit")
            body += LEVEL2.pack(level, width)
            array = rows.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            if not np.isfinite(array).all():
                raise ValueError("rows cannot be represented in wire float32")
            body += array.astype("<f4", copy=False).tobytes(order="C")
        return bytes(body) + hmac.digest(self.key, body, "sha256")

    def decode(self, frame: bytes) -> Publication:
        if not HEADER2.size + 32 <= len(frame) <= self.max_frame:
            raise ValueError("invalid frame size")
        body, signature = frame[:-32], frame[-32:]
        if not hmac.compare_digest(signature, hmac.digest(self.key, body, "sha256")):
            raise ValueError("frame authentication failed")
        magic, version, writer, session, fingerprint, epoch, seq, start, tokens, count = HEADER2.unpack_from(body)
        if magic != MAGIC2 or version != 2:
            raise ValueError("unknown protocol/version")
        if not 1 <= tokens <= MAX_TOKENS or not 1 <= count <= MAX_LEVELS:
            raise ValueError("unsafe frame dimensions")
        offset, levels = HEADER2.size, {}
        for _ in range(count):
            if offset + LEVEL2.size > len(body):
                raise ValueError("truncated descriptor")
            level, width = LEVEL2.unpack_from(body, offset)
            offset += LEVEL2.size
            if level in levels or not 1 <= width <= MAX_WIDTH:
                raise ValueError("duplicate level or unsafe width")
            nbytes = tokens * width * 4
            if offset + nbytes > len(body):
                raise ValueError("truncated rows")
            array = np.frombuffer(body, dtype="<f4", count=tokens * width, offset=offset).copy()
            levels[level] = torch.from_numpy(array.astype(np.float32, copy=False).reshape(tokens, width))
            offset += nbytes
        if offset != len(body):
            raise ValueError("trailing untyped data is forbidden")
        pub = Publication(UUID(bytes=session), writer, fingerprint.hex(), epoch, seq, start, levels)
        pub.check()
        return pub
