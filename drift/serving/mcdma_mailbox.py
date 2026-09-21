"""Verified single-slot mailboxes over one-sided MCDMA reads and writes."""
from __future__ import annotations
import mmap
import os
import struct
import zlib

TARGET_MAGIC, MAGIC, ACK_MAGIC, RECEIPT_MAGIC = b"DRFTTGT1", b"DRFTMB02", b"DRFTAK02", b"DRFTRC01"
TARGET = struct.Struct("<8sQQ")            # magic, session, region size
HEADER = struct.Struct("<8sQQIII")         # magic, session, sequence, length, crc32, state
ACK = struct.Struct("<8sQQI")              # magic, session, sequence, status (1 consumed, 2 rejected)
RECEIPT = struct.Struct("<8sQII")           # magic, session, length, crc32; then the receipt's bytes (a DIFFERENT fact from ACK, see write_receipt)
COMMITTED, TARGET_AT, HEADER_AT, ACK_AT, RECEIPT_AT, RECEIPT_MAX, PAYLOAD_AT = 0xC0117ED, 0, 256, 2048, 4096, 4000, 8192
CHUNK = 8192                               # one wire segment is 8,960 payload bytes: records and chunks are single placements


class MailboxError(RuntimeError):
    pass


class MailboxTransportError(MailboxError):
    """A retryable transport failure with the protocol state still retained."""


class MappedRegion:
    """The consumer's direct view of the target's region file; same put/get surface as a remote connection."""
    def __init__(self, path: str):
        fd = os.open(path, os.O_RDWR)
        try:
            self.region_len = os.fstat(fd).st_size
            self.memory = mmap.mmap(fd, self.region_len, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)

    def put(self, data, offset: int = 0) -> None:
        self.memory[offset:offset + len(data)] = data

    def get(self, length: int, offset: int = 0) -> bytes:
        return bytes(self.memory[offset:offset + length])


def _span(conn, offset: int, length: int) -> int:
    if offset < 0 or length < 0 or offset + length > int(conn.region_len):
        raise MailboxError("access outside the region")
    return offset


RETRIES = {"count": 0, "events": []}                     # the client library is plain UDP request/response: a lost datagram is a timeout, not a retransmit


def _again(call, what=""):
    """put and get are idempotent here (same bytes, same offset), so a transport timeout is retried; bounds and protocol errors are not."""
    import time
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            return call()
        except MailboxError:
            raise
        except Exception as error:
            native_timeout = type(error).__name__ == "MCDMAError" and str(error) in {
                "get failed (rc=-1)", "put failed (rc=-1)", "put failed (rc=-2)"}
            if not isinstance(error, (TimeoutError, ConnectionError, BlockingIOError)) and not native_timeout:
                raise MailboxError("non-retryable mailbox transport failure") from error
            RETRIES["events"] = (RETRIES["events"] + [f"{what} attempt {attempt} {type(error).__name__}: {error} after {time.perf_counter() - t0:.3f}s"])[-12:]
            if attempt == 2:
                raise MailboxTransportError("mailbox transport retries exhausted") from error
            RETRIES["count"] += 1


def _put(conn, offset, data):
    for a in range(0, len(data), CHUNK):
        part = data[a:a + CHUNK]
        at = _span(conn, offset + a, len(part))
        _again(lambda: conn.put(part, offset=at), f"put {len(part)}@{at}")


def _safe_read(n: int) -> int:
    """Widen reads around the known USB-C loss band without crossing the region."""
    return n + (-n) % 1024 if 640 <= n % 1024 <= 712 else n


def _get(conn, offset, length):
    out = []
    for a in range(0, length, CHUNK):
        n = min(CHUNK, length - a)
        ask = _safe_read(n)
        _span(conn, offset + a, n)                                             # the request ITSELF must lie inside the region
        if offset + a + ask > int(conn.region_len):                            # widening would leave the region: read an earlier, wider window instead
            back = offset + a + ask - int(conn.region_len)
            at = _span(conn, offset + a - back, ask)
            part = bytes(_again(lambda: conn.get(ask, offset=at), f"get {ask}@{at}"))[back:back + n]
        else:
            at = _span(conn, offset + a, ask)
            part = bytes(_again(lambda: conn.get(ask, offset=at), f"get {ask}@{at}"))[:n]
        if len(part) != n:
            raise MailboxError("short read from the region")
        out.append(part)
    return b"".join(out)


class Window:
    """A sub-range [base, base + size) of a region with the put/get surface of a connection: one region, several mailboxes."""
    def __init__(self, conn, base: int, size: int):
        if base < 0 or size <= 0 or base % 4096 or base + size > int(conn.region_len):
            raise MailboxError("window outside the region")
        self.conn, self.base, self.region_len = conn, base, size

    def put(self, data, offset: int = 0) -> None:
        self.conn.put(data, offset=self.base + _span(self, offset, len(data)))

    def get(self, length: int, offset: int = 0) -> bytes:
        return self.conn.get(length, offset=self.base + _span(self, offset, length))


class Reader:
    """Consume one mailbox; construction creates a new session and clears its slot."""
    def __init__(self, conn, session: int | None = None):
        if int(conn.region_len) < PAYLOAD_AT + 1:
            raise MailboxError("region too small for a mailbox")
        self.conn, self.session, self.expected = conn, session or int.from_bytes(os.urandom(8), "little") | 1, 1
        _put(conn, HEADER_AT, bytes(HEADER.size)); _put(conn, ACK_AT, bytes(ACK.size))
        _put(conn, TARGET_AT, TARGET.pack(TARGET_MAGIC, self.session, int(conn.region_len)))

    def peek(self) -> bytes | None:
        """The next committed publication of THIS session with its CRC verified, NOT yet acknowledged; None if nothing new."""
        header = _get(self.conn, HEADER_AT, HEADER.size)
        magic, session, sequence, length, crc, state = HEADER.unpack(header)
        if magic != MAGIC or state != COMMITTED or session != self.session or sequence < self.expected:
            return None                                                        # empty, uncommitted, another session's leftovers, or already consumed
        if sequence != self.expected or not 0 < length <= int(self.conn.region_len) - PAYLOAD_AT:
            _put(self.conn, ACK_AT, ACK.pack(ACK_MAGIC, self.session, sequence, 2))
            raise MailboxError("publication out of sequence or out of bounds")
        payload = _get(self.conn, PAYLOAD_AT, length)
        if _get(self.conn, HEADER_AT, HEADER.size) != header or zlib.crc32(payload) != crc:
            return None                                                        # torn or being rewritten: not acknowledged, the writer still owns it
        return payload

    def acknowledge(self, consumed: bool = True) -> None:
        """Tell the producer what happened to the publication peek() returned; only `consumed` frees the slot for reuse."""
        _put(self.conn, ACK_AT, ACK.pack(ACK_MAGIC, self.session, self.expected, 1 if consumed else 2))
        self.expected += 1

    def poll(self) -> bytes | None:
        payload = self.peek()
        if payload is not None:
            self.acknowledge(True)
        return payload


class Writer:
    """Remote producer. Adopts the consumer's session at open; refuses a region without a valid TARGET record."""
    def __init__(self, conn):
        magic, session, size = TARGET.unpack(_get(conn, TARGET_AT, TARGET.size))
        if magic != TARGET_MAGIC or session == 0 or size != int(conn.region_len):
            raise MailboxError("this region is not a live Drift mailbox target")
        self.conn, self.session, self.sequence, self.pending = conn, session, 0, None
        if any(_get(conn, HEADER_AT, HEADER.size)) or any(_get(conn, ACK_AT, ACK.size)):
            raise MailboxError("writer restart requires a fresh consumer session")
        self.last_ack = 0
        self._publishing = None
        self._failed = False

    def _check_session(self):
        record = TARGET.unpack(_get(self.conn, TARGET_AT, TARGET.size))
        if record != (TARGET_MAGIC, self.session, int(self.conn.region_len)):
            raise MailboxError("consumer restarted: session changed, publication lost")

    def acknowledged(self) -> bool:
        self._check_session()
        if self.pending is None:
            return self.sequence > 0 and self.last_ack == self.sequence
        magic, session, sequence, status = ACK.unpack(_get(self.conn, ACK_AT, ACK.size))
        if magic == ACK_MAGIC and session == self.session and sequence == self.pending:
            if status != 1:
                raise MailboxError("consumer rejected the publication")
            self.last_ack = self.pending
            self.pending = None
        return self.pending is None

    def publish(self, payload: bytes) -> int:
        self._check_session()
        if self._failed:
            raise MailboxError("writer failed; a fresh consumer session is required")
        if not 0 < len(payload) <= int(self.conn.region_len) - PAYLOAD_AT:
            raise MailboxError("payload does not fit the region")
        payload = bytes(payload)
        if self._publishing is not None:
            if payload != self._publishing:
                raise MailboxError("an incomplete publication must be retried with identical bytes")
            if self.acknowledged():
                self._publishing = None
                return self.sequence
        else:
            if self.pending is not None and not self.acknowledged():
                raise MailboxError("previous publication not acknowledged; the slot cannot be reused")
            self.sequence += 1
            self.pending, self._publishing = self.sequence, payload
        crc = zlib.crc32(payload)
        _put(self.conn, HEADER_AT, HEADER.pack(MAGIC, self.session, self.sequence, 0, 0, 0))       # retract the previous header first
        _put(self.conn, PAYLOAD_AT, payload)
        if zlib.crc32(_get(self.conn, PAYLOAD_AT, len(payload))) != crc:                           # write completion VERIFIED before the commit marker
            self._failed = True
            raise MailboxError("payload read-back does not match; nothing was committed")
        self._check_session()
        _put(self.conn, HEADER_AT, HEADER.pack(MAGIC, self.session, self.sequence, len(payload), crc, COMMITTED))
        self._publishing = None
        return self.sequence


def write_receipt(conn, session: int, data: bytes) -> None:
    """Relay cache-application receipts separately from inbox-delivery acknowledgements."""
    if not 0 < len(data) <= RECEIPT_MAX:
        raise MailboxError("receipt does not fit its slot")
    _put(conn, RECEIPT_AT, RECEIPT.pack(RECEIPT_MAGIC, session, 0, 0))                              # retract, body, then the header last
    _put(conn, RECEIPT_AT + RECEIPT.size, data)
    _put(conn, RECEIPT_AT, RECEIPT.pack(RECEIPT_MAGIC, session, len(data), zlib.crc32(data)))


def read_receipt(conn, session: int) -> bytes | None:
    """Producer side: the latest receipt of THIS session, or None (absent, another session's, or torn)."""
    magic, seen, length, crc = RECEIPT.unpack(_get(conn, RECEIPT_AT, RECEIPT.size))
    if magic != RECEIPT_MAGIC or seen != session or not 0 < length <= RECEIPT_MAX:
        return None
    data = _get(conn, RECEIPT_AT + RECEIPT.size, length)
    return data if zlib.crc32(data) == crc else None
