"""Writer lifetime and bridge retry regressions without native transport."""
import pytest

from drift.serving.mcdma_mailbox import MailboxError, Reader, Writer
from test_mcdma_mailbox import FakeRegion, _bridge_module


def test_new_writer_cannot_claim_an_acknowledgement():
    region = FakeRegion()
    Reader(region)
    assert not Writer(region).acknowledged()


def test_writer_rebuild_cannot_reuse_a_consumed_sequence():
    region = FakeRegion()
    reader = Reader(region)
    writer = Writer(region)
    writer.publish(b"first")
    assert reader.poll() == b"first"
    assert writer.acknowledged()
    with pytest.raises(MailboxError, match="fresh.*session"):
        Writer(region)


def test_ack_checks_current_session_even_when_old_ack_matches():
    from drift.serving.mcdma_mailbox import ACK, ACK_AT
    region = FakeRegion()
    reader = Reader(region, session=11)
    writer = Writer(region)
    writer.publish(b"first")
    reader.poll()
    old_ack = region.get(ACK.size, ACK_AT)
    Reader(region, session=13)
    region.put(old_ack, ACK_AT)
    with pytest.raises(MailboxError, match="session changed"):
        writer.acknowledged()


def test_bridge_never_rebuilds_after_protocol_failure(tmp_path):
    module = _bridge_module()
    reverse, forward = FakeRegion(), FakeRegion()
    bridge = module.Bridge(Reader(reverse), reverse, str(tmp_path / "in"),
                           str(tmp_path / "out"), 0, True, forward_conn=forward)
    reader = Reader(forward)
    folder = tmp_path / "out" / "s"
    folder.mkdir(parents=True)
    for index in range(2):
        (folder / f"{index:06d}.npz").write_bytes(b"tap")
    bridge.handle(module.pack("watch", "s", 0))
    bridge.publish_taps()
    writer = bridge.forward
    reader.acknowledge(False)
    bridge.serve(0.01, lambda _: None)
    assert bridge.forward is writer
    assert bridge.in_flight == 0
    assert bridge.next_tap == 1


class TimeoutRegion(FakeRegion):
    timeouts = 0

    def get(self, length, offset=0):
        if self.timeouts:
            self.timeouts -= 1
            raise TimeoutError("reply never arrives")
        return super().get(length, offset)


def test_bridge_survives_exhausted_read_retries(tmp_path):
    module = _bridge_module()
    region = TimeoutRegion()
    reader = Reader(region)
    writer = Writer(region)
    writer.publish(module.pack("stage", "s", 0, b"data"))
    bridge = module.Bridge(reader, region, str(tmp_path / "in"), str(tmp_path / "out"), 1, False)
    region.timeouts = 5
    assert bridge.serve(0.02, lambda _: None) == 1
    assert writer.acknowledged()
    assert (tmp_path / "in/s/000000.npz").read_bytes() == b"data"


def test_partial_publish_retry_keeps_the_same_sequence():
    region = TimeoutRegion()
    reader = Reader(region)
    writer = Writer(region)
    original_put = region.put
    def lose_readback(data, offset=0):
        from drift.serving.mcdma_mailbox import PAYLOAD_AT
        original_put(data, offset)
        if offset == PAYLOAD_AT:
            region.timeouts = 3
    region.put = lose_readback
    with pytest.raises(Exception):
        writer.publish(b"first")
    region.put = original_put
    assert writer.publish(b"first") == 1
    assert reader.poll() == b"first"
    assert writer.acknowledged()


def test_ack_timeout_does_not_repeat_release(tmp_path):
    from drift.serving.mcdma_mailbox import ACK_AT
    module = _bridge_module()
    region = FakeRegion()
    reader = Reader(region)
    writer = Writer(region)
    bridge = module.Bridge(reader, region, str(tmp_path / "in"), str(tmp_path / "out"), 0, True)
    bridge.handle(module.pack("stage", "s", 0, b"data"))
    writer.publish(module.pack("release", "s", 0))
    real_put = region.put
    failures = [5]
    def fail_ack(data, offset=0):
        if offset == ACK_AT and failures[0]:
            failures[0] -= 1
            raise TimeoutError("ack lost")
        real_put(data, offset)
    region.put = fail_ack
    assert bridge.serve(0.02, lambda _: None) == 1
    assert writer.acknowledged()
    assert bridge.waiting == [("s", 0)]


def test_lost_commit_reply_cannot_duplicate_an_already_consumed_payload():
    from drift.serving.mcdma_mailbox import COMMITTED, HEADER, HEADER_AT, MailboxTransportError
    region = FakeRegion()
    reader = Reader(region)
    writer = Writer(region)
    real_put = region.put
    received = []
    def lose_commit_reply(data, offset=0):
        real_put(data, offset)
        if offset == HEADER_AT and HEADER.unpack(data)[-1] == COMMITTED:
            payload = reader.poll()
            if payload is not None:
                received.append(payload)
            raise TimeoutError("commit landed but reply lost")
    region.put = lose_commit_reply
    with pytest.raises(MailboxTransportError):
        writer.publish(b"once")
    region.put = real_put
    assert writer.publish(b"once") == 1
    assert reader.poll() is None
    assert writer.acknowledged()
    assert received == [b"once"]
    assert writer.publish(b"next") == 2
    assert reader.poll() == b"next"


def test_forward_bridge_keeps_unacknowledged_tap_through_timeouts(tmp_path):
    module = _bridge_module()
    reverse, forward = FakeRegion(), TimeoutRegion()
    reader = Reader(forward)
    bridge = module.Bridge(Reader(reverse), reverse, str(tmp_path / "in"), str(tmp_path / "out"),
                           0, True, forward_conn=forward)
    folder = tmp_path / "out/s"
    folder.mkdir(parents=True)
    (folder / "000000.npz").write_bytes(b"tap")
    bridge.handle(module.pack("watch", "s", 0))
    bridge.publish_taps()
    writer = bridge.forward
    forward.timeouts = 5
    bridge.serve(0.02, lambda _: None)
    assert bridge.forward is writer and bridge.in_flight == 0
    assert not bridge.forward_failed
    assert reader.poll().endswith(b"\ntap")
    bridge.publish_taps()
    assert bridge.in_flight is None and bridge.next_tap == 1
