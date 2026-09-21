"""Validate and apply forward taps before acknowledging their mailbox slot."""
import io
import json
import math
from dataclasses import dataclass
from zipfile import ZipFile

import numpy as np

from drift.serving.mcdma_mailbox import COMMITTED, HEADER, HEADER_AT, MAGIC, MailboxError, _get
from drift.exchange.lifetime import CheckedRegion


MAX_BYTES = 32 << 20


def require(condition, message):
    if not condition:
        raise MailboxError(message)


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate forward envelope field")
        result[key] = value
    return result


def decode(payload, session, expected, layers, previous_stop):
    head, delimiter, body = payload.partition(b"\n")
    require(delimiter and len(head) <= 4096 and len(body) <= MAX_BYTES, "invalid forward envelope size")
    meta = json.loads(head, object_pairs_hook=unique_fields)
    if type(meta) is dict and meta.get("op") == "complete":
        require(set(meta) == {"op", "session", "tap_count", "bytes", "source_start", "source_stop"}, "invalid forward completion fields")
        require(meta["session"] == session and type(meta["tap_count"]) is int and meta["tap_count"] == expected, "forward completion session or count mismatch")
        require(type(meta["bytes"]) is int and meta["bytes"] == 0 and body == b"", "invalid forward completion body")
        require(type(meta['source_start']) is int and type(meta['source_stop']) is int
                and 0 <= meta['source_start'] <= meta['source_stop'], 'invalid forward completion positions')
        require(meta['source_stop'] == (previous_stop if previous_stop is not None else meta['source_start']),
                'forward completion does not cover final source frontier')
        return Complete(expected, meta['source_start'], meta['source_stop'])
    require(type(meta) is dict and set(meta) == {"session", "tap", "bytes", "file_mtime_ns"}, "invalid forward envelope fields")
    require(meta["session"] == session, "forward session mismatch")
    require(type(meta["tap"]) is int and meta["tap"] == expected, "forward tap sequence mismatch")
    require(type(meta["bytes"]) is int and meta["bytes"] == len(body), "forward byte count mismatch")
    require(type(meta["file_mtime_ns"]) is int and meta["file_mtime_ns"] > 0, "invalid forward timestamp")
    names = {f"l{layer}" for layer in layers} | {"start", "stop"}
    arrays = {}
    with ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        require(len(members) == len(names) and {m.filename for m in members} == {n + ".npy" for n in names}, "forward layer set mismatch")
        require(sum(m.file_size for m in members) <= MAX_BYTES, "forward archive expands beyond limit")
        for member in members:
            name = member.filename[:-4]
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                require(version in ((1, 0), (2, 0)), "unsupported forward array version")
                read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = read_header(source)
                if name in ("start", "stop"):
                    require(shape == () and dtype.kind in "iu", "invalid forward position")
                else:
                    require(len(shape) == 2 and 0 < shape[0] <= 4096 and shape[1] == 512 and dtype.kind == "f", "invalid forward latent shape or dtype")
                require(math.prod(shape) * dtype.itemsize == member.file_size - source.tell(), "forward array byte count mismatch")
            with archive.open(member) as source:
                arrays[name] = np.lib.format.read_array(source, allow_pickle=False)
    start, stop = int(arrays.pop("start")), int(arrays.pop("stop"))
    require(0 <= start < stop and (previous_stop is None or start == previous_stop), "forward positions are not contiguous")
    latents = {}
    for layer in layers:
        with np.errstate(over="ignore", invalid="ignore"):
            value = arrays[f"l{layer}"].astype(np.float32)
        require(value.shape == (stop - start, 512) and np.isfinite(value).all(), "invalid or nonfinite forward latents")
        latents[layer] = value
    return Tap(meta, latents, start, stop)


@dataclass(frozen=True)
class Tap:
    meta: dict
    latents: dict
    start: int
    stop: int


@dataclass(frozen=True)
class Complete:
    tap_count: int
    source_start: int
    source_stop: int


class ForwardMailbox:
    def __init__(self, reader, session, layers):
        if not isinstance(reader.conn, CheckedRegion):
            reader.conn = CheckedRegion(reader.conn)
        self.reader, self.session, self.layers = reader, session, tuple(layers)
        self.expected, self.previous_stop, self.pending, self.poisoned = 0, None, None, False
        self.finished = False
        self.first_start = None

    def peek(self):
        require(not self.poisoned, "forward session poisoned; start a new session")
        require(self.pending is None, "forward publication already pending")
        try:
            raw = _get(self.reader.conn, HEADER_AT, HEADER.size)
            magic, session, sequence, _, _, state = HEADER.unpack(raw)
            if state != COMMITTED:
                return None
            require(magic == MAGIC and session == self.reader.session, "stale or malformed committed forward mailbox")
            if sequence == self.reader.expected - 1:
                return None
            require(not self.finished, "forward publication after completion")
            require(sequence == self.reader.expected, "forward mailbox sequence mismatch")
            payload = self.reader.peek()
            require(payload is not None, "committed forward mailbox is corrupt or changed")
            self.pending = decode(payload, self.session, self.expected, self.layers, self.previous_stop)
            if isinstance(self.pending, Complete) and self.first_start is not None:
                require(self.pending.source_start == self.first_start, 'forward completion source start mismatch')
            return self.pending
        except Exception:
            self.poisoned = True
            raise

    def acknowledge(self, tap):
        require(not self.poisoned and tap is self.pending, "forward acknowledgement without validated delivery")
        try:
            self.reader.acknowledge(True)
        except Exception:
            self.poisoned = True
            raise
        if isinstance(tap, Complete):
            self.finished = True
        else:
            if self.first_start is None:
                self.first_start = tap.start
            self.previous_stop, self.expected = tap.stop, self.expected + 1
        self.pending = None


def validate_translation(entries, keys, order, source_rows, layers, index_dim):
    order = np.asarray(order)
    require(order.ndim == 1 and order.size > 0 and order.dtype.kind in "iu", "invalid translated forward row order")
    require((order >= 0).all() and (order < source_rows).all() and (order[1:] >= order[:-1]).all(), "translated forward row order out of bounds")
    require(type(entries) is dict and set(entries) == set(layers), "translated forward layer set mismatch")
    require(type(keys) is dict and set(keys) == set(layers), "translated forward selector layer set mismatch")
    rows = len(order)
    for layer in layers:
        pair = entries[layer]
        require(isinstance(pair, (tuple, list)) and len(pair) == 2, "invalid translated forward KV pair")
        for value, shape in ((pair[0], (rows, 2, 256)), (pair[1], (rows, 2, 256)), (keys[layer], (rows, index_dim))):
            array = np.asarray(value)
            require(array.shape == shape and array.dtype.kind == "f" and np.isfinite(array).all(), "invalid or nonfinite translated forward entries")
    return rows
