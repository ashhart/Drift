"""Drive the reusable exchange core with fake runtimes: no model, host, MCDMA or activation channel."""
import hashlib
import pytest

from drift.exchange.session import ExchangeSession, ExchangeError


class Source:
    def __init__(self, rows=1):
        self.rows, self.calls = rows, 0

    def snapshot(self, copies):
        self.calls += 1
        return dict(body=b'x' * 64, sha256=hashlib.sha256(b'x' * 64).hexdigest(), rows=self.rows)


class Link:
    def __init__(self, ranks=('spark-a.invalid', 'spark-b.invalid')):
        self.ranks, self.delivered, self.queue, self.acked = ranks, [], [], []

    def deliver(self, session, sequence, body, copies):
        self.delivered.append((session, sequence, len(body), copies))
        return dict(sha256='a' * 64, sequence=sequence)

    def confirm_applied(self, delivered, rows):
        return dict(ranks=tuple(self.ranks), receipts=tuple('b' * 64 for _ in self.ranks))

    def peek(self):
        return self.queue.pop(0) if self.queue else None

    def acknowledge(self, tap):
        self.acked.append(tap)


class Sink:
    def __init__(self):
        self.applied = []

    def prepare(self, tap):
        return tap['rows'], tap

    def commit(self, prepared):
        self.applied.append(prepared)
        return prepared['rows']


def build(mode='drift', **kwargs):
    return ExchangeSession(session='live-1', source_worker='qwen', target_worker='glm', mode=mode,
                           ranks=('spark-a.invalid', 'spark-b.invalid'), source=Source(), link=Link(), sink=Sink(), copies=12, **kwargs)


def test_drift_publication_returns_a_validated_record_and_advances_the_sequence():
    exchange = build()
    first = exchange.publish_own()
    assert first['sequence'] == 0 and first['rows'] == 12 and first['mode'] == 'drift'
    assert first['applied_ranks'] == ('spark-a.invalid', 'spark-b.invalid') and first['text_bytes'] == 0
    assert exchange.publish_own()['sequence'] == 1


def test_text_mode_publishes_nothing_and_records_the_text_it_carried():
    exchange = build(mode='text')
    record = exchange.publish_own(text_bytes=256)
    assert record['rows'] == 0 and record['bytes'] == 0 and record['text_bytes'] == 256
    assert record['applied_ranks'] == () and exchange.link.delivered == []


def test_drift_mode_refuses_to_carry_text():
    exchange = build()
    with pytest.raises(ExchangeError):
        exchange.publish_own(text_bytes=1)


def test_a_rank_that_does_not_confirm_poisons_the_session():
    exchange = build()
    exchange.link.confirm_applied = lambda delivered, rows: dict(ranks=('spark-a.invalid',), receipts=('b' * 64,))
    with pytest.raises(ExchangeError):
        exchange.publish_own()
    assert exchange.poisoned
    with pytest.raises(ExchangeError):
        exchange.publish_own()


def test_forward_drain_applies_each_tap_once_and_acknowledges_it():
    exchange = build()
    exchange.link.queue = [dict(rows=3, start=0, stop=3), dict(rows=2, start=3, stop=5)]
    applied = exchange.drain_forward()
    assert [record['rows'] for record in applied] == [3, 2]
    assert len(exchange.link.acked) == 2 and exchange.foreign_rows == 5


def test_text_mode_drains_nothing_from_the_link():
    exchange = build(mode='text')
    exchange.link.queue = [dict(rows=3, start=0, stop=3)]
    assert exchange.drain_forward() == []
    assert exchange.link.acked == [] and exchange.foreign_rows == 0


def test_the_foreign_row_cap_poisons_rather_than_silently_dropping():
    exchange = build(row_cap=4)
    exchange.link.queue = [dict(rows=3, start=0, stop=3), dict(rows=2, start=3, stop=5)]
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.poisoned


def test_an_unknown_mode_is_refused_at_construction():
    with pytest.raises(ExchangeError):
        build(mode='hybrid')


def test_the_foreign_cap_refuses_before_any_memory_is_applied():
    exchange = build(row_cap=4)
    exchange.link.queue = [dict(rows=3, start=0, stop=3), dict(rows=2, start=3, stop=5)]
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.poisoned
    assert [tap['rows'] for tap in exchange.sink.applied] == [3]
    assert exchange.foreign_rows == 3


def test_a_tap_over_the_cap_is_never_acknowledged():
    exchange = build(row_cap=4)
    exchange.link.queue = [dict(rows=5, start=0, stop=5)]
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.sink.applied == [] and exchange.link.acked == []


def test_an_exchange_error_from_the_runtime_still_poisons_the_session():
    for collaborator, method in (('source', 'snapshot'), ('link', 'deliver'), ('link', 'confirm_applied')):
        exchange = build()
        def raise_exchange(*_args, **_kwargs):
            raise ExchangeError('EXCHANGE_RUNTIME')
        setattr(getattr(exchange, collaborator), method, raise_exchange)
        with pytest.raises(ExchangeError):
            exchange.publish_own()
        assert exchange.poisoned, f'{collaborator}.{method} escaped without poisoning'


def test_an_exchange_error_from_the_sink_poisons_the_session():
    exchange = build()
    def raise_exchange(*_args, **_kwargs):
        raise ExchangeError('EXCHANGE_RUNTIME')
    exchange.sink.prepare = raise_exchange
    exchange.link.queue = [dict(rows=1, start=0, stop=1)]
    with pytest.raises(ExchangeError):
        exchange.drain_forward()
    assert exchange.poisoned
