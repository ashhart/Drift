"""Reverse Drift publications over the dedicated MCDMA targets: stage on every rank, release on the head, then bind each
rank's connector receipt (cache application) to the same live session, sequence and payload digest.

Two different confirmations, never merged: `staged` / `released` are mailbox acknowledgements ("this rank holds these bytes in
its inbox"); `applied` needs the GLM connector's own per-rank receipt relayed by the bridge ("this rank's cache holds them")."""
from __future__ import annotations
import hashlib
from contextvars import copy_context
import json
import threading
import time
from drift.serving.mcdma_mailbox import MailboxError, Writer, read_receipt
from drift.exchange.lifetime import CheckedRegion, checkpoint


def envelope(op: str, session: str, sequence: int, body: bytes = b"", copies: int = 1) -> bytes:
    return json.dumps({"op": op, "session": session, "sequence": sequence, "bytes": len(body), "copies": copies}).encode() + b"\n" + body


class ReversePublisher:
    def __init__(self, connections: dict, head: str, timeout_s: float = 30.0, split: int | None = None):
        """connections: rank name -> one-sided connection to that rank's dedicated target; `head` names the scheduler's rank."""
        if head not in connections:
            raise ValueError("the head rank must be one of the connections")
        from drift.serving.mcdma_mailbox import Window
        self.raw, self.head, self.timeout_s = connections, head, timeout_s
        self.conns = {rank: CheckedRegion(conn if split is None else Window(conn, 0, split))
                      for rank, conn in connections.items()}
        self.writers, self._partial = {rank: Writer(conn) for rank, conn in self.conns.items()}, {}

    def _deliver(self, rank: str, payload: bytes) -> None:
        checkpoint()
        writer, deadline = self.writers[rank], time.monotonic() + self.timeout_s
        writer.publish(payload)
        while not writer.acknowledged():
            checkpoint()
            if time.monotonic() > deadline:
                raise MailboxError(f"{rank}: no acknowledgement in time")

    def publish(self, session: str, sequence: int, body: bytes, expected_rows: int | None = None, copies: int = 1) -> dict:
        """deliver() then confirm_applied(): use the two halves separately when the cache write can only happen after the
        caller starts or resumes the receiving request."""
        delivered = self.deliver(session, sequence, body, copies)
        return {**delivered, **self.confirm_applied(delivered, expected_rows)}

    def watch(self, session: str) -> None:
        """Ask the head's bridge to send this live session's decode-time taps back through the forward mailbox."""
        self._deliver(self.head, envelope("watch", session, 0))

    def deliver(self, session: str, sequence: int, body: bytes, copies: int = 1) -> dict:
        """`copies` > 1: each rank's bridge tiles the rows locally, so only one copy crosses the wire."""
        digest, t0, errors = hashlib.sha256(body).hexdigest(), time.perf_counter(), {}

        def stage(rank):
            try:
                self._deliver(rank, envelope("stage", session, sequence, body, copies))
            except Exception as error:
                errors[rank] = error
        threads = [threading.Thread(target=copy_context().run, args=(stage, rank)) for rank in self.conns]
        for t in threads: t.start()
        for t in threads: t.join()
        if errors:
            detail = "; ".join(f"{rank}: {type(e).__name__}: {e}" for rank, e in sorted(errors.items()))
            raise MailboxError(f"staging failed, nothing was released ({detail})"[:280])     # the head's file stays hidden: no rank applies a partial delivery
        checkpoint()
        staged = time.perf_counter() - t0
        self._deliver(self.head, envelope("release", session, sequence))
        released = time.perf_counter() - t0
        return {"session": session, "sequence": sequence, "bytes": len(body), "sha256": digest, "staged_all_ranks_s": round(staged, 6), "released_s": round(released, 6), "_t0": time.perf_counter()}

    def confirm_applied(self, delivered: dict, expected_rows: int | None = None) -> dict:
        """Block until every rank's connector receipt for this publication is bound, or fail."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            checkpoint()
            result = self.poll_applied(delivered, expected_rows)
            if result is not None:
                return result
            if time.monotonic() > deadline:
                missing = sorted(set(self.conns) - set(self._seen(delivered)))
                raise MailboxError(f"no connector receipt from {missing}: delivered to the inbox, NOT confirmed in the cache")

    def _seen(self, delivered: dict) -> dict:
        return self._partial.setdefault((delivered["session"], delivered["sequence"]), {})

    def poll_applied(self, delivered: dict, expected_rows: int | None = None) -> dict | None:
        """Non-blocking: bind whatever receipts have arrived; the result once EVERY rank's cache applied this publication."""
        session, sequence, digest, found = delivered["session"], delivered["sequence"], delivered["sha256"], self._seen(delivered)
        for rank, conn in self.conns.items():
            if rank in found:
                continue
            raw = read_receipt(conn, self.writers[rank].session)
            if raw is None:
                continue
            for relayed in json.loads(raw).get("receipts", []):
                receipt = relayed.get("connector_receipt", {})
                if relayed.get("live_session") != session or receipt.get("sequence") != sequence:
                    continue
                index = list(self.conns).index(rank)
                wrong_rank = (receipt.get("rank"), receipt.get("world_size")) != (index, len(self.conns))
                # chain: what was published (source) -> what this rank's bridge wrote (staged, tiled there) -> what this rank's cache applied
                if wrong_rank or relayed.get("source_sha256") != digest or receipt.get("sha256") != relayed.get("staged_sha256") or (expected_rows is not None and receipt.get("rows") != expected_rows):
                    raise MailboxError(f"{rank}: the cache applied something other than what was published")
                found[rank] = {**receipt, "source_sha256": relayed["source_sha256"], "receipt_mtime_ns": relayed.get("receipt_mtime_ns"), "seen_after_s": round(time.perf_counter() - delivered["_t0"], 6)}
        if len(found) < len(self.conns):
            return None
        if len({r["sha256"] for r in found.values()}) != 1:
            raise MailboxError("ranks applied different files")
        return {"applied_all_ranks_s": round(max(r["seen_after_s"] for r in found.values()), 6), "receipts": dict(found)}
