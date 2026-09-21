"""Exercise request cancellation inside real mailbox reads and rank staging."""
import threading

import pytest

from drift.exchange.lifetime import request_scope
from drift.exchange.session import ExchangeError
from drift.serving.mcdma_forward import ForwardMailbox
from drift.serving.mcdma_mailbox import MailboxError, PAYLOAD_AT, Reader, Writer
from drift.serving.mcdma_reverse import ReversePublisher
from tests.test_mcdma_forward import GL, publication
from tests.test_mcdma_mailbox import FakeRegion


def checker(cancelled):
    def check():
        if cancelled.is_set():
            raise ExchangeError('EXCHANGE_CANCELLED')
    return check


def test_rank_staging_inherits_cancellation_and_never_commits_after_a_cancelled_chunk():
    cancelled = threading.Event()

    class Region(FakeRegion):
        def put(self, data, offset=0):
            super().put(data, offset)
            if offset == PAYLOAD_AT:
                cancelled.set()

    region = Region()
    reader = Reader(region)
    publisher = ReversePublisher({'rank-0': region}, head='rank-0', timeout_s=.01)
    with request_scope(checker(cancelled)), pytest.raises(MailboxError):
        publisher.deliver('live-1', 0, bytes(32 << 10))
    assert cancelled.is_set() and reader.peek() is None
    assert [offset for offset, _ in region.puts if offset >= PAYLOAD_AT] == [PAYLOAD_AT]


def test_forward_read_cancellation_never_returns_or_acknowledges_the_tap():
    cancelled = threading.Event()

    class Region(FakeRegion):
        armed = False

        def get(self, length, offset=0):
            result = super().get(length, offset)
            if self.armed and offset == PAYLOAD_AT:
                cancelled.set()
            return result

    region = Region()
    reader = Reader(region)
    forward = ForwardMailbox(reader, 'live-1', GL)
    writer = Writer(region)
    writer.publish(publication(session='live-1', start=0, stop=2))
    region.armed = True
    with request_scope(checker(cancelled)), pytest.raises(MailboxError):
        forward.peek()
    assert forward.poisoned and not writer.acknowledged()
