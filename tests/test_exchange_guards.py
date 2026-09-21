"""Malformed collaborator results must not mutate caches or leave reusable sessions."""
import hashlib

import pytest

from drift.exchange.session import ExchangeError
from test_exchange_session import build


@pytest.mark.parametrize("field,value", [("rows", 0), ("rows", True), ("sha256", "bad"), ("body", "text")])
def test_snapshot_is_checked_before_delivery(field, value):
    exchange = build()
    snapshot = dict(body=b"cache", rows=1, sha256=hashlib.sha256(b"cache").hexdigest())
    snapshot[field] = value
    exchange.source.snapshot = lambda _: snapshot
    with pytest.raises(ExchangeError):
        exchange.publish_own()
    assert exchange.link.delivered == []
    assert exchange.poisoned and exchange.sequence == 0


def test_malformed_receipt_result_poisons_before_another_publication():
    exchange = build()
    exchange.link.confirm_applied = lambda *_: {}
    with pytest.raises(ExchangeError):
        exchange.publish_own()
    assert exchange.poisoned
    with pytest.raises(ExchangeError, match="POISONED"):
        exchange.publish_own()
    assert len(exchange.link.delivered) == 1


def test_missing_tap_frontier_cannot_be_applied_or_acknowledged():
    exchange = build()
    exchange.link.queue = [dict(rows=1, start=0)]
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.sink.applied == [] and exchange.link.acked == []
    assert exchange.poisoned and exchange.foreign_rows == 0


def test_outgoing_capacity_is_checked_before_delivery():
    exchange = build(publish_row_cap=12)
    exchange.publish_own()
    with pytest.raises(ExchangeError, match="PUBLISH_CAP"):
        exchange.publish_own()
    assert len(exchange.link.delivered) == 1 and exchange.poisoned
