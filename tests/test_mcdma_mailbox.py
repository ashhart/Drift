"""Reverse-direction mailbox: target identity, bounds, session, verified write before the commit marker, acknowledgement before reuse."""
import pytest
from drift.serving.mcdma_mailbox import ACK, ACK_AT, ACK_MAGIC, COMMITTED, HEADER, HEADER_AT, MAGIC, MailboxError, MappedRegion, PAYLOAD_AT, Reader, Writer


class FakeRegion:
    def __init__(self, length=1 << 20):
        self.memory, self.region_len, self.puts, self.drop = bytearray(length), length, [], None

    def put(self, data, offset=0):
        assert 0 <= offset and offset + len(data) <= self.region_len
        self.puts.append((offset, len(data)))
        if self.drop is not None and offset == self.drop:
            return                                                              # a write that never landed
        self.memory[offset:offset + len(data)] = data

    def get(self, length, offset=0):
        assert 0 <= offset and offset + length <= self.region_len
        return bytes(self.memory[offset:offset + length])


def test_publish_consume_acknowledge_and_no_reuse_before_the_ack():
    region = FakeRegion(); reader = Reader(region); writer = Writer(region)
    assert reader.poll() is None
    payload = bytes(range(256)) * 600
    assert writer.publish(payload) == 1 and region.puts[-1] == (HEADER_AT, HEADER.size)          # the committed header is the LAST write
    with pytest.raises(MailboxError, match="not acknowledged"):
        writer.publish(b"next")
    assert reader.poll() == payload and reader.poll() is None
    assert writer.acknowledged() and writer.publish(b"next") == 2 and reader.poll() == b"next"


def test_a_region_without_a_live_target_record_is_refused():
    with pytest.raises(MailboxError, match="not a live Drift mailbox"):
        Writer(FakeRegion())                                                    # e.g. some other daemon answered on this address
    small, big = FakeRegion(1 << 20), FakeRegion(2 << 20)
    Reader(small); big.memory[:64] = small.memory[:64]                          # a copied record whose size does not match this region
    with pytest.raises(MailboxError, match="not a live Drift mailbox"):
        Writer(big)


def test_stale_acknowledgements_and_publications_from_another_session_are_rejected():
    region = FakeRegion(); reader = Reader(region, session=11); writer = Writer(region)
    writer.publish(b"one")
    region.memory[ACK_AT:ACK_AT + ACK.size] = ACK.pack(ACK_MAGIC, 99, 1, 1)     # right sequence, WRONG session
    assert not writer.acknowledged()
    restarted = Reader(region, session=12)                                      # consumer restart: new session, slot cleared
    assert restarted.poll() is None
    with pytest.raises(MailboxError, match="session changed"):
        writer.acknowledged()
    region.memory[HEADER_AT:HEADER_AT + HEADER.size] = HEADER.pack(MAGIC, 11, 1, 3, 0, COMMITTED)   # the old session's header reappears
    assert restarted.poll() is None


def test_a_payload_that_did_not_land_is_never_committed_and_corruption_is_never_consumed():
    region = FakeRegion(); reader = Reader(region); writer = Writer(region)
    region.drop = PAYLOAD_AT
    with pytest.raises(MailboxError, match="read-back"):
        writer.publish(b"lost in flight")
    assert HEADER.unpack(region.get(HEADER.size, HEADER_AT))[5] != COMMITTED and reader.poll() is None
    region.drop = None
    reader = Reader(region)
    writer = Writer(region)
    writer.publish(b"hello world")
    region.memory[PAYLOAD_AT] ^= 0xFF
    assert reader.poll() is None and not writer.acknowledged()
    region.memory[PAYLOAD_AT] ^= 0xFF
    region.memory[HEADER_AT:HEADER_AT + HEADER.size] = HEADER.pack(MAGIC, reader.session, 5, 11, 0, COMMITTED)
    with pytest.raises(MailboxError, match="out of sequence"):
        reader.poll()
    Reader(region)
    with pytest.raises(MailboxError, match="does not fit"):
        Writer(region).publish(bytes(1 << 20))


def test_the_consumer_maps_the_region_file_directly(tmp_path):
    path = tmp_path / "region.bin"; path.write_bytes(bytes(1 << 16))
    consumer_view, producer_view = MappedRegion(str(path)), MappedRegion(str(path))
    reader = Reader(consumer_view); writer = Writer(producer_view)
    writer.publish(b"through the shared mapping")
    assert reader.poll() == b"through the shared mapping" and writer.acknowledged()
    with pytest.raises(MailboxError, match="outside the region"):
        from drift.serving.mcdma_mailbox import _get
        _get(producer_view, (1 << 16) - 4, 8)


def _bridge_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("spark_bridge", "scripts/mcdma_target/spark_bridge.py"); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _run(bridge, seconds=0.25):
    import threading
    worker = threading.Thread(target=bridge.serve, args=(seconds, lambda line: None)); worker.start(); worker.join()


def test_head_stages_hidden_then_releases_and_the_connector_receipt_is_relayed_separately_from_the_ack(tmp_path):
    import hashlib, json
    from drift.serving.mcdma_mailbox import read_receipt
    module = _bridge_module()
    region = FakeRegion(); reader = Reader(region); writer = Writer(region)
    inbox, outbox = tmp_path / "in", tmp_path / "out"
    bridge = module.Bridge(reader, region, str(inbox), str(outbox), rank=0, head=True)
    body = bytes(range(200)) * 50
    writer.publish(module.pack("loop-1", 0, body) if False else module.pack("stage", "loop-1", 0, body))
    assert reader.peek() is not None and not writer.acknowledged()              # seen, not yet acknowledged
    _run(bridge)
    assert writer.acknowledged() and (inbox / "loop-1" / ".000000.tmp.npz").read_bytes() == body and not (inbox / "loop-1" / "000000.npz").exists()
    assert read_receipt(region, writer.session) is None                         # delivered to the inbox is NOT applied to the cache
    writer.publish(module.pack("release", "loop-1", 0))
    _run(bridge)
    assert writer.acknowledged() and (inbox / "loop-1" / "000000.npz").read_bytes() == body and read_receipt(region, writer.session) is None
    (outbox / "loop-1").mkdir(parents=True)
    connector = {"rank": 0, "world_size": 2, "sequence": 0, "rows": 96, "sha256": hashlib.sha256(body).hexdigest()}
    (outbox / "loop-1" / "ack.000000.rank0.json").write_text(json.dumps(connector))
    _run(bridge)
    relayed = json.loads(read_receipt(region, writer.session))["receipts"][-1]
    assert type(relayed.pop("receipt_mtime_ns")) is int                          # the Spark's own clock, for the causal schedule
    assert relayed == {"live_session": "loop-1", "connector_receipt": connector, "source_sha256": hashlib.sha256(body).hexdigest(), "staged_sha256": hashlib.sha256(body).hexdigest(), "copies": 1}
    assert read_receipt(region, writer.session + 2) is None                     # another session never sees it


def test_a_worker_rank_writes_the_visible_file_and_bad_envelopes_are_rejected_not_delivered(tmp_path):
    module = _bridge_module()
    region = FakeRegion(); reader = Reader(region); writer = Writer(region)
    bridge = module.Bridge(reader, region, str(tmp_path / "in"), str(tmp_path / "out"), rank=1, head=False)
    writer.publish(module.pack("stage", "loop-1", 3, b"abc")); _run(bridge)
    assert writer.acknowledged() and (tmp_path / "in" / "loop-1" / "000003.npz").read_bytes() == b"abc"
    for bad in (module.pack("release", "loop-1", 3), b'{"op": "stage", "session": "../escape", "sequence": 0, "bytes": 1}\nx', b'{"op": "stage", "session": "s", "sequence": 0, "bytes": 5}\nx'):
        writer.publish(bad); _run(bridge)
        with pytest.raises(MailboxError, match="rejected"):
            writer.acknowledged()
        writer.pending = None
    assert not (tmp_path / "escape").exists() and sorted(x.name for x in (tmp_path / "in" / "loop-1").iterdir()) == ["000003.npz"]


def test_reverse_publisher_binds_both_ranks_receipts_to_session_sequence_and_digest(tmp_path):
    import hashlib, json, threading, time
    from drift.serving.mcdma_reverse import ReversePublisher
    module = _bridge_module()
    regions = {"spark-a.invalid": FakeRegion(), "spark-b.invalid": FakeRegion()}
    bridges = {}
    for n, (rank, region) in enumerate(regions.items()):
        bridges[rank] = module.Bridge(Reader(region), region, str(tmp_path / rank / "in"), str(tmp_path / rank / "out"), rank=n, head=rank == "spark-a.invalid")
    body = b"publication" * 1000

    def connector(wrong=False):                                                 # stands in for the GLM connector: applies only once the HEAD's file is visible
        head_file = tmp_path / "spark-a.invalid" / "in" / "live-7" / "000002.npz"
        while not head_file.exists():
            time.sleep(0.005)
        assert (tmp_path / "spark-b.invalid" / "in" / "live-7" / "000002.npz").exists()    # the other rank already held the file when the head released
        for n, rank in enumerate(regions):
            out = tmp_path / rank / "out" / "live-7"; out.mkdir(parents=True)
            (out / f"ack.000002.rank{n}.json").write_text(json.dumps({"rank": n, "world_size": 2, "sequence": 2, "rows": 96, "sha256": hashlib.sha256(b"other" if wrong and n == 1 else body).hexdigest()}))

    workers = [threading.Thread(target=b.serve, args=(2.0, lambda line: None)) for b in bridges.values()] + [threading.Thread(target=connector)]
    for w in workers: w.start()
    result = ReversePublisher(regions, head="spark-a.invalid", timeout_s=3).publish("live-7", 2, body, expected_rows=96)
    for w in workers: w.join()
    assert result["staged_all_ranks_s"] <= result["released_s"] and result["applied_all_ranks_s"] >= 0 and set(result["receipts"]) == {"spark-a.invalid", "spark-b.invalid"}
    assert all(r["sha256"] == result["sha256"] == r["source_sha256"] and r["sequence"] == 2 for r in result["receipts"].values())


def test_reverse_publisher_refuses_a_cache_that_applied_different_bytes_and_reports_missing_application(tmp_path):
    import hashlib, json, threading
    from drift.serving.mcdma_reverse import ReversePublisher
    module = _bridge_module()
    region = FakeRegion(); bridge = module.Bridge(Reader(region), region, str(tmp_path / "in"), str(tmp_path / "out"), rank=0, head=True)
    worker = threading.Thread(target=bridge.serve, args=(1.5, lambda line: None)); worker.start()
    publisher = ReversePublisher({"spark-a.invalid": region}, head="spark-a.invalid", timeout_s=0.5)
    with pytest.raises(MailboxError, match="NOT confirmed in the cache"):
        publisher.publish("live-8", 0, b"abc")
    (tmp_path / "out" / "live-8").mkdir(parents=True)
    (tmp_path / "out" / "live-8" / "ack.000001.rank0.json").write_text(json.dumps({"rank": 0, "world_size": 1, "sequence": 1, "rows": 1, "sha256": hashlib.sha256(b"different").hexdigest()}))
    with pytest.raises(MailboxError, match="other than what was published"):
        publisher.publish("live-8", 1, b"abc")
    worker.join()


def test_reads_never_use_the_size_band_that_the_usb_link_drops():
    class Lossy(FakeRegion):
        def get(self, length, offset=0):
            if 675 <= length % 1024 <= 678:
                raise TimeoutError("reply never arrives")                       # what both USB-C legs do today
            return super().get(length, offset)
    region = Lossy(); reader = Reader(region); writer = Writer(region)
    for size in (675, 2723, 8192 + 676, 3 * 8192 + 1699):
        payload = bytes([size % 251]) * size
        writer.publish(payload)
        assert reader.poll() == payload and writer.acknowledged()
    from drift.serving.mcdma_mailbox import _get, _safe_read
    assert _safe_read(2723) == 3072 and _safe_read(2000) == 2000 and _safe_read(8192) == 8192
    region.memory[-700:] = bytes(range(250)) * 2 + bytes(200)
    assert _get(region, region.region_len - 676, 676) == bytes(region.memory[-676:])      # at the very end of the region the window moves back instead


def test_tiling_on_the_rank_keeps_the_chain_source_to_staged_to_applied(tmp_path):
    import hashlib, io, json, threading, time
    import numpy as np
    from drift.serving.mcdma_reverse import ReversePublisher
    module = _bridge_module()
    import sys; sys.path.insert(0, "scripts/mcdma_target")
    regions = {"spark-a.invalid": FakeRegion(4 << 20), "spark-b.invalid": FakeRegion(4 << 20)}
    bridges = [module.Bridge(Reader(region), region, str(tmp_path / rank / "in"), str(tmp_path / rank / "out"), rank=n, head=n == 0) for n, (rank, region) in enumerate(regions.items())]
    sink = io.BytesIO(); np.savez(sink, l3=np.arange(5 * 512, dtype=np.float16).reshape(5, 512)); body = sink.getvalue()

    def connector():
        head_file = tmp_path / "spark-a.invalid" / "in" / "s" / "000000.npz"
        while not head_file.exists():
            time.sleep(0.005)
        for n, rank in enumerate(regions):
            data = (tmp_path / rank / "in" / "s" / "000000.npz").read_bytes()
            assert np.load(io.BytesIO(data))["l3"].shape == (60, 512)
            out = tmp_path / rank / "out" / "s"; out.mkdir(parents=True)
            (out / f"ack.000000.rank{n}.json").write_text(json.dumps({"rank": n, "world_size": 2, "sequence": 0, "rows": 60, "sha256": hashlib.sha256(data).hexdigest()}))

    workers = [threading.Thread(target=b.serve, args=(2.0, lambda line: None)) for b in bridges] + [threading.Thread(target=connector)]
    for w in workers: w.start()
    result = ReversePublisher(regions, head="spark-a.invalid", timeout_s=3).publish("s", 0, body, expected_rows=60, copies=12)
    for w in workers: w.join()
    assert result["bytes"] == len(body) and all(r["source_sha256"] == result["sha256"] and r["rows"] == 60 for r in result["receipts"].values())
    assert len({r["sha256"] for r in result["receipts"].values()}) == 1 and next(iter(result["receipts"].values()))["sha256"] != result["sha256"]


def test_two_mailboxes_share_one_region_without_touching_each_other():
    from drift.serving.mcdma_mailbox import Window
    region = FakeRegion(4 << 20)
    reverse, forward = Window(region, 0, 2 << 20), Window(region, 2 << 20, 2 << 20)
    r_reader, f_reader = Reader(reverse), Reader(forward)                       # reverse consumed on the Spark, forward consumed from the Studio
    r_writer, f_writer = Writer(reverse), Writer(forward)
    r_writer.publish(b"qwen to glm" * 100); f_writer.publish(b"glm to qwen" * 3000)
    assert f_reader.poll() == b"glm to qwen" * 3000 and r_reader.poll() == b"qwen to glm" * 100
    assert r_writer.acknowledged() and f_writer.acknowledged() and r_writer.session != f_writer.session
    with pytest.raises(MailboxError, match="outside the region"):
        forward.get(16, offset=(2 << 20) - 8)
    with pytest.raises(MailboxError, match="window outside"):
        Window(region, 3 << 20, 2 << 20)
    assert bytes(region.memory[(2 << 20) - 64:(2 << 20)]) == bytes(64)          # nothing of the reverse mailbox leaked across the boundary... its payload ends far below


def test_the_head_sends_decode_time_taps_back_in_order_once_told_which_session_to_watch(tmp_path):
    import json, threading, time
    from drift.serving.mcdma_mailbox import Window
    module = _bridge_module()
    region = FakeRegion(4 << 20)
    reverse, forward = Window(region, 0, 2 << 20), Window(region, 2 << 20, 2 << 20)
    bridge = module.Bridge(Reader(reverse), reverse, str(tmp_path / "in"), str(tmp_path / "out"), rank=0, head=True, forward_conn=forward, log_path=str(tmp_path / "bridge.jsonl"))
    studio_side = Reader(forward)                                               # the reader host stamps the forward window before anything is sent
    out = tmp_path / "out" / "live-3"; out.mkdir(parents=True)
    (out / "000000.npz").write_bytes(b"tap zero" * 500); (out / "000001.npz").write_bytes(b"tap one" * 700)
    worker = threading.Thread(target=bridge.serve, args=(1.5, lambda line: None)); worker.start()
    assert studio_side.poll() is None                                           # nothing is sent until the producer is told which session to watch
    writer = Writer(reverse); writer.publish(module.pack("watch", "live-3", 0))
    got, deadline = [], time.time() + 1.2
    while len(got) < 2 and time.time() < deadline:
        payload = studio_side.poll()
        if payload is not None:
            head, _, body = payload.partition(b"\n"); got.append((json.loads(head), body))
    worker.join()
    assert [g[0]["tap"] for g in got] == [0, 1] and got[0][1] == b"tap zero" * 500 and got[1][1] == b"tap one" * 700
    assert all(g[0]["session"] == "live-3" and type(g[0]["file_mtime_ns"]) is int for g in got)
    events = [json.loads(l)["event"] for l in (tmp_path / "bridge.jsonl").read_text().splitlines()]
    assert events[:3] == ["tap_published", "tap_acknowledged", "tap_published"]   # strictly one in flight
