"""Stage MCDMA publications, relay rank receipts and publish ordered forward taps."""
import argparse, fcntl, hashlib, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mcdma_mailbox import MailboxError, MailboxTransportError, MappedRegion, Reader, Window, Writer, write_receipt
except ImportError:
    from drift.serving.mcdma_mailbox import MailboxError, MailboxTransportError, MappedRegion, Reader, Window, Writer, write_receipt
NAME = re.compile(r"[A-Za-z0-9_.-]{1,96}")


def acquire_region(path):
    """Hold an exclusive lease before a reader can stamp the shared region."""
    lease = open(path, "rb")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lease.close()
        raise
    return lease


def pack(op: str, session: str, sequence: int, body: bytes = b"", copies: int = 1) -> bytes:
    return json.dumps({"op": op, "session": session, "sequence": sequence, "bytes": len(body), "copies": copies}).encode() + b"\n" + body


def unpack(payload: bytes):
    head, _, body = payload.partition(b"\n")
    meta = json.loads(head)
    meta.setdefault("copies", 1) if isinstance(meta, dict) else None
    if not isinstance(meta, dict) or set(meta) != {"op", "session", "sequence", "bytes", "copies"} or type(meta["copies"]) is not int or not 1 <= meta["copies"] <= 64 \
            or meta["op"] not in ("stage", "release", "watch") or type(meta["session"]) is not str \
            or not NAME.fullmatch(meta["session"]) or type(meta["sequence"]) is not int or not 0 <= meta["sequence"] < 10**6 or meta["bytes"] != len(body) \
            or (meta["op"] in ("release", "watch")) != (len(body) == 0):
        raise ValueError("invalid publication envelope")
    return meta["op"], meta["session"], meta["sequence"], body, meta["copies"]


class Bridge:
    def __init__(self, reader, conn, inbox: str, outbox: str, rank: int, head: bool, forward_conn=None, log_path: str | None = None):
        self.reader, self.conn, self.inbox, self.outbox, self.rank, self.head = reader, conn, inbox, outbox, rank, head
        self.staged, self.waiting = {}, []                                      # (session, sequence) -> digests; receipts still to relay
        self.forward_conn, self.forward, self.watching, self.next_tap, self.in_flight = forward_conn, None, None, 0, None
        self.log_path = log_path
        self.forward_failed = False
        self.forward_complete = False
        self.watched = set()
        self._pending_ack = None

    def _log(self, **event) -> None:
        if self.log_path:
            with open(self.log_path, "a") as sink:
                sink.write(json.dumps(event) + "\n")

    def publish_taps(self) -> None:
        """Publish watched taps in order, keeping one unacknowledged publication."""
        if self.forward_conn is None or self.watching is None or self.forward_failed:
            return
        if self.forward is None:
            try:
                self.forward = Writer(self.forward_conn)                        # the remote consumer stamps the forward window first
            except MailboxError:
                return
        if self.in_flight is not None:
            if not self.forward.acknowledged():
                return
            if self.in_flight == "complete":
                self._log(event="forward_complete_acknowledged", tap_count=self.next_tap, ns=time.time_ns())
            else:
                self._log(event="tap_acknowledged", tap=self.in_flight, ns=time.time_ns())
            self.in_flight = None
        if self.forward_complete:
            return
        path = os.path.join(self.outbox, self.watching, f"{self.next_tap:06d}.npz")
        if os.path.exists(path):
            with open(path, "rb") as source:
                body = source.read()
            self.forward.publish(json.dumps({"session": self.watching, "tap": self.next_tap, "bytes": len(body), "file_mtime_ns": os.stat(path).st_mtime_ns}).encode() + b"\n" + body)
            self._log(event="tap_published", tap=self.next_tap, bytes=len(body), file_mtime_ns=os.stat(path).st_mtime_ns, ns=time.time_ns())
            self.in_flight, self.next_tap = self.next_tap, self.next_tap + 1
            return
        self._publish_completion()

    def _publish_completion(self):
        folder = os.path.join(self.outbox, self.watching)
        finished = os.path.join(folder, "finished")
        if not os.path.isfile(finished):
            return
        names = os.listdir(folder)
        if any(name.startswith(".") and name.endswith(".tmp.npz") for name in names):
            return
        if os.path.isfile(os.path.join(folder, f"{self.next_tap:06d}.npz")):
            return
        if any(re.fullmatch(r"[0-9]{6}\.npz", name) and int(name[:6]) >= self.next_tap for name in names):
            raise MailboxError("connector tap sequence has a gap at completion")
        with open(finished) as source:
            try:
                outcome = json.load(source)
            except ValueError:
                return
        if (type(outcome) is not dict or set(outcome) != {"failed", "writes_scheduled", "tap_count", "source_start", "source_stop"}
                or outcome["failed"] != "" or type(outcome["writes_scheduled"]) is not int
                or outcome["writes_scheduled"] < 0
                or type(outcome["tap_count"]) is not int or outcome["tap_count"] != self.next_tap
                or type(outcome["source_start"]) is not int or type(outcome["source_stop"]) is not int
                or not 0 <= outcome["source_start"] <= outcome["source_stop"]
                or any(name.startswith("error.rank") for name in names)):
            raise MailboxError("connector finished with a failed or invalid session")
        self.forward.publish(json.dumps({"op": "complete", "session": self.watching,
                                         "tap_count": self.next_tap, "bytes": 0,
                                         "source_start": outcome["source_start"], "source_stop": outcome["source_stop"]}).encode() + b"\n")
        self.forward_complete, self.in_flight = True, "complete"
        self._log(event="forward_complete_published", tap_count=self.next_tap, ns=time.time_ns())

    def _paths(self, session, sequence):
        folder = os.path.join(self.inbox, session)
        return folder, os.path.join(folder, f"{sequence:06d}.npz"), os.path.join(folder, f".{sequence:06d}.tmp.npz")

    def handle(self, payload: bytes) -> None:
        op, session, sequence, body, copies = unpack(payload)
        folder, final, hidden = self._paths(session, sequence)
        if op == "watch":                                                       # which live session's taps to send back; head only
            if not self.head:
                raise ValueError("taps exist on the head host only")
            if self.watching is not None:
                if session == self.watching and not self.forward_failed:
                    return
                if (self.forward_failed or not self.forward_complete or self.forward is None
                        or not self.forward.acknowledged() or session in self.watched):
                    raise ValueError("the previous forward stream must complete before changing requests")
            if len(self.watched) >= 1024:
                raise ValueError("forward request budget exhausted")
            self.watched.add(session)
            self.watching = session
            self.next_tap, self.in_flight, self.forward_complete = 0, None, False
            return
        if op == "stage":
            source = hashlib.sha256(body).hexdigest()
            if copies > 1:                                                     # tile HERE: the repeated rows never cross the wire
                from npz_tile import tile
                body, _ = tile(body, copies)
            os.makedirs(folder, exist_ok=True)
            partial = hidden + ".part"
            with open(partial, "wb") as sink:
                sink.write(body)
            os.replace(partial, hidden if self.head else final)
            self.staged[(session, sequence)] = {"source_sha256": source, "staged_sha256": hashlib.sha256(body).hexdigest(), "copies": copies}
            if not self.head:
                self.waiting.append((session, sequence))
        else:
            if not self.head or (session, sequence) not in self.staged:
                raise ValueError("release without a staged file on the head host")
            os.replace(hidden, final)
            self._log(event="released", session=session, sequence=sequence, ns=time.time_ns())
            self.waiting.append((session, sequence))

    def relay_receipts(self) -> None:
        for key in list(self.waiting):
            session, sequence = key
            path = os.path.join(self.outbox, session, f"ack.{sequence:06d}.rank{self.rank}.json")
            if os.path.exists(path):
                try:
                    with open(path) as source:
                        receipt = json.load(source)
                except (OSError, ValueError):
                    continue                                                    # still being written: look again on the next pass, never die on it
                history = (getattr(self, "history", []) + [{"live_session": session, "connector_receipt": receipt, "receipt_mtime_ns": os.stat(path).st_mtime_ns, **self.staged.get(key, {})}])[-8:]
                write_receipt(self.conn, self.reader.session, json.dumps({"receipts": history}).encode())
                self.history = history
                self._log(event="receipt_relayed", session=session, sequence=sequence, receipt_mtime_ns=os.stat(path).st_mtime_ns, ns=time.time_ns())
                self.waiting.remove(key)

    def serve(self, seconds: float, log=print) -> int:
        deadline, handled = time.monotonic() + seconds, 0
        while time.monotonic() < deadline:
            try:
                self.relay_receipts()
            except MailboxTransportError:
                log(json.dumps({"receipt_retry": True}))
            try:
                self.publish_taps()
            except MailboxTransportError:
                log(json.dumps({"forward_retry": True}))
            except MailboxError as error:
                self.forward_failed = True
                log(json.dumps({"forward_failed": str(error), "fresh_session_required": True}))
            try:
                if self._pending_ack is None:
                    payload = self.reader.peek()
                    if payload is None:
                        time.sleep(0.00005)
                        continue
                    try:
                        self.handle(payload)
                        self._pending_ack = True
                    except Exception as error:
                        self._pending_ack = False
                        log(json.dumps({"failed": type(error).__name__}))
                self.reader.acknowledge(self._pending_ack)
                handled += int(self._pending_ack)
                self._pending_ack = None
            except MailboxTransportError:
                log(json.dumps({"reverse_retry": True}))
                time.sleep(0.001)
            except MailboxError as error:
                log(json.dumps({"rejected": str(error)}))
                break
        return handled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="/dev/shm/drift-mailbox/region.bin")
    parser.add_argument("--inbox", default="/dev/shm/glm53-handoff/tp-live-in")
    parser.add_argument("--outbox", default="/dev/shm/glm53-handoff/tp-live-out")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--log", help="jsonl of tap and receipt events with this host's clock")
    args = parser.parse_args()
    with acquire_region(args.region):
        region = MappedRegion(args.region)
        half = region.region_len // 2                                           # Lower half receives; upper half sends on the head only.
        reverse = Window(region, 0, half); reader = Reader(reverse)
        print(json.dumps({"session": reader.session, "rank": args.rank, "head": args.head}), flush=True)
        bridge = Bridge(reader, reverse, args.inbox, args.outbox, args.rank, args.head, Window(region, half, half) if args.head else None, args.log)
        print(json.dumps({"handled": bridge.serve(args.seconds, lambda line: print(line, flush=True))}), flush=True)
