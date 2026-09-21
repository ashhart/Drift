"""Compose real mailbox decoding and live adapters with synthetic translated tensors."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from drift.exchange.live import ForeignRowSink, MailboxLink
from drift.exchange.session import ExchangeError
from drift.serving.mcdma_forward import ForwardMailbox
from drift.serving.mcdma_mailbox import Reader, Writer
from tests.test_exchange_live import Bank
from tests.test_exchange_session import build
from tests.test_mcdma_forward import GL, Region, publication


@pytest.fixture
def route(monkeypatch):
    region = Region()
    mailbox = Reader(region, session=7)
    writer = Writer(region)
    events = []

    def translate(latents):
        rows = len(latents[GL[0]])
        entries = {0: (np.ones((rows, 2, 256)), np.ones((rows, 2, 256)))}
        return entries, {0: np.ones((rows, 128))}, np.arange(rows), None

    def append(*args, **kwargs):
        assert not writer.acknowledged()
        events.append('append')
        return 0

    monkeypatch.setattr('drift.exchange.live.append_entries', append)
    sink = ForeignRowSink({}, None, (0,), SimpleNamespace(read=translate), Bank(), 128,
                          lambda: 10, evaluate=lambda: events.append('settled'))
    exchange = build()
    exchange.link = MailboxLink(None, ForwardMailbox(mailbox, 'live-1', GL))
    exchange.sink = sink
    return exchange, writer, events


def finish(writer, count, start, stop):
    writer.publish(json.dumps(dict(op='complete', session='live-1', tap_count=count,
                                   bytes=0, source_start=start, source_stop=stop)).encode() + b'\n')


def test_typed_taps_apply_once_then_complete_without_mutating_cache(route):
    exchange, writer, events = route
    for index in range(2):
        writer.publish(publication(session='live-1', tap=index, start=20 + index * 2, stop=22 + index * 2))
        assert exchange.drain_forward()[0]['rows'] == 2
        assert writer.acknowledged()
    assert events == ['append', 'settled', 'append', 'settled']
    finish(writer, 2, 20, 24)
    assert exchange.drain_forward() == []
    assert writer.acknowledged() and exchange.forward_complete
    assert exchange.foreign_rows == 4 and exchange.source_stop == 24
    assert exchange.drain_forward() == [] and len(events) == 4
    writer.publish(publication(session='live-1', tap=2, start=24, stop=26))
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.poisoned and not writer.acknowledged() and len(events) == 4


def test_zero_tap_completion_never_calls_the_sink(route):
    exchange, writer, events = route
    finish(writer, 0, 0, 0)
    assert exchange.drain_forward() == []
    assert writer.acknowledged() and exchange.forward_complete and events == []


def test_actual_tap_over_capacity_stays_unacknowledged_and_unapplied(route):
    exchange, writer, events = route
    exchange.row_cap = 1
    writer.publish(publication(session='live-1', start=0, stop=2))
    with pytest.raises(ExchangeError, match='EXCHANGE_FOREIGN_CAP'):
        exchange.drain_forward()
    assert exchange.poisoned and not writer.acknowledged() and events == []


def test_incorrect_completion_frontier_is_not_acknowledged(route):
    exchange, writer, events = route
    writer.publish(publication(session='live-1', start=0, stop=2))
    exchange.drain_forward()
    finish(writer, 1, 0, 3)
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.poisoned and not writer.acknowledged()
    assert events == ['append', 'settled']
