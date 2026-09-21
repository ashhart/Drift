from __future__ import annotations
import hashlib
import hmac
import struct
from uuid import UUID
import numpy as np
import torch
from drift.core.types import Delta, KV

# Big-endian structural integers; little-endian float32 tensor payload.
HEADER = struct.Struct("!8sBB16sQQQIH")
LAYER = struct.Struct("!HHH")
MAGIC = b"TELEKV01"
MAX_FRAME = 64 * 1024 * 1024
MAX_TOKENS, MAX_LAYERS, MAX_HEADS, MAX_DIM = 8192, 128, 128, 1024


class FrameCodec:
    """CPU/copied reference format. No pickle, JSON payload, strings, or token IDs.

    HMAC authenticates bytes; it does NOT encrypt or prove absence of covert
    channels. Session policy and sequence/layer validation are additional checks.
    Production GPU formats need new versioned dtype/packing contracts.
    """
    def __init__(self, key: bytes, max_frame: int = MAX_FRAME):
        if len(key) < 32 or not 1024 <= max_frame <= MAX_FRAME:
            raise ValueError("use >=32-byte secret and bounded frame capacity")
        self.key, self.max_frame = key, max_frame

    def encode(self, delta: Delta) -> bytes:
        delta.check()
        if not 1 <= delta.tokens <= MAX_TOKENS or len(delta.layers) > MAX_LAYERS:
            raise ValueError("frame dimension limit exceeded")
        size = HEADER.size + 32 + sum(LAYER.size + kv.k.numel() * 8 for kv in delta.layers.values())
        if size > self.max_frame:
            raise ValueError("frame too large; chunk the publication")
        body = bytearray(HEADER.pack(MAGIC, 1, delta.direction, delta.session.bytes,
                                     delta.epoch, delta.sequence, delta.start,
                                     delta.tokens, len(delta.layers)))
        for index, kv in sorted(delta.layers.items()):
            _, heads, dim = kv.k.shape
            if heads > MAX_HEADS or dim > MAX_DIM:
                raise ValueError("head shape exceeds wire limit")
            body += LAYER.pack(index, heads, dim)
            for tensor in (kv.k, kv.v):
                array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                if not np.isfinite(array).all():
                    raise ValueError("activation cannot be represented in wire float32")
                body += array.astype("<f4", copy=False).tobytes(order="C")
        return bytes(body) + hmac.digest(self.key, body, "sha256")

    def decode(self, frame: bytes) -> Delta:
        if not HEADER.size + 32 <= len(frame) <= self.max_frame:
            raise ValueError("invalid frame size")
        body, signature = frame[:-32], frame[-32:]
        if not hmac.compare_digest(signature, hmac.digest(self.key, body, "sha256")):
            raise ValueError("frame authentication failed")
        magic, version, direction, session, epoch, seq, start, tokens, count = HEADER.unpack_from(body)
        if magic != MAGIC or version != 1 or direction not in (0, 1):
            raise ValueError("unknown protocol/version/direction")
        if not 1 <= tokens <= MAX_TOKENS or not 1 <= count <= MAX_LAYERS:
            raise ValueError("unsafe frame dimensions")
        offset, layers = HEADER.size, {}
        for _ in range(count):
            if offset + LAYER.size > len(body):
                raise ValueError("truncated descriptor")
            index, heads, dim = LAYER.unpack_from(body, offset)
            offset += LAYER.size
            if index in layers or not 1 <= heads <= MAX_HEADS or not 1 <= dim <= MAX_DIM:
                raise ValueError("duplicate layer or unsafe shape")
            nbytes = tokens * heads * dim * 4
            if offset + 2 * nbytes > len(body):
                raise ValueError("truncated tensor")
            arrays = []
            for _ in range(2):
                # copy: the decoded tensor must not alias a reusable socket buffer.
                array = np.frombuffer(body, dtype="<f4", count=tokens * heads * dim,
                                      offset=offset).copy().astype(np.float32, copy=False)
                arrays.append(torch.from_numpy(array.reshape(tokens, heads, dim)))
                offset += nbytes
            layers[index] = KV(*arrays)
        if offset != len(body):
            raise ValueError("trailing untyped data is forbidden")
        delta = Delta(UUID(bytes=session), direction, epoch, seq, start, layers)
        delta.check()
        return delta
