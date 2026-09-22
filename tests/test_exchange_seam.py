"""Compose actual mailbox tap types with the exchange adapters using synthetic IO."""
import numpy as np
import pytest

from drift.exchange.live import ForeignRowSink, LiveExchangeError, MailboxLink
from drift.exchange.session import ExchangeError, ExchangeSession
from drift.serving.mcdma_forward import Complete, Tap


class Forward:
    """Yields exactly what ForwardMailbox.peek yields: Tap and Complete dataclasses, never dicts."""

    def __init__(self, items):
        self.items, self.acked = list(items), []

    def peek(self):
        return self.items[0] if self.items else None

    def acknowledge(self, native):
        assert native is self.items[0], 'the mailbox identifies its pending tap by identity'
        self.acked.append(self.items.pop(0))


class Publisher:
    def deliver(self, session, sequence, body, copies):
        return dict(session=session, sequence=sequence, sha256='a' * 64)

    def confirm_applied(self, delivered, rows):
        return dict(receipts={'rank-b': {'sha256': 'c' * 64}, 'rank-a': {'sha256': 'b' * 64}})


class Source:
    def snapshot(self, copies):
        body = b'publication'
        import hashlib
        return dict(body=body, sha256=hashlib.sha256(body).hexdigest(), rows=1)


class Reader:
    def read(self, latents):
        rows = len(next(iter(latents.values())))
        return {0: (np.ones((rows, 2), np.float32), np.ones((rows, 2), np.float32))}, None, list(range(rows)), None


class Bank:
    def positions(self, sources, own):
        return np.asarray(sources, dtype=np.int64)

    def remember(self, entries, sources, first_slot):
        pass


def tap(start, stop):
    return Tap(meta={'tap': start}, latents={0: np.ones((stop - start, 2))}, start=start, stop=stop)


def build(items, monkeypatch, ranks=('rank-a', 'rank-b')):
    import drift.exchange.live as live
    monkeypatch.setattr(live, 'validate_translation', lambda entries, keys, order, rows, layers, dim: rows)
    monkeypatch.setattr(live, 'append_entries', lambda *a, **k: 0)
    forward = Forward(items)
    sink = ForeignRowSink(None, None, (0,), Reader(), Bank(), 4, lambda: 0)
    session = ExchangeSession(session='live-1', source_worker='qwen', target_worker='glm', mode='drift',
                              ranks=ranks, source=Source(), link=MailboxLink(Publisher(), forward),
                              sink=sink, copies=12)
    return session, forward


def test_real_mailbox_taps_drain_without_poisoning(monkeypatch):
    session, forward = build([tap(0, 2), tap(2, 5)], monkeypatch)
    applied = session.drain_forward()
    assert [item['rows'] for item in applied] == [2, 3]
    assert session.foreign_rows == 5 and not session.poisoned
    assert [type(item).__name__ for item in forward.acked] == ['Tap', 'Tap']


def test_a_real_completion_marker_finishes_the_stream(monkeypatch):
    session, forward = build([tap(0, 2), Complete(tap_count=1, source_start=0, source_stop=2)], monkeypatch)
    assert len(session.drain_forward()) == 1
    assert session.forward_complete and not session.poisoned
    assert [type(item).__name__ for item in forward.acked] == ['Tap', 'Complete']


def test_a_tap_after_completion_still_poisons(monkeypatch):
    session, _ = build([Complete(tap_count=0, source_start=0, source_stop=0), tap(0, 2)], monkeypatch)
    session.drain_forward()
    with pytest.raises(ExchangeError):
        session.drain_forward()
    assert session.poisoned


def test_a_gap_in_the_real_tap_sequence_poisons(monkeypatch):
    session, _ = build([tap(0, 2), tap(3, 5)], monkeypatch)
    with pytest.raises(ExchangeError):
        session.drain_forward()
    assert session.poisoned


def test_ranks_configured_head_first_still_validate(monkeypatch):
    session, _ = build([], monkeypatch, ranks=('rank-b', 'rank-a'))
    record = session.publish_own()
    assert record['applied_ranks'] == ('rank-a', 'rank-b') and not session.poisoned


def test_peek_normalises_both_kinds_while_retaining_the_message_to_acknowledge():
    native = tap(0, 1)
    link = MailboxLink(Publisher(), Forward([native]))
    normalised = link.peek()
    assert normalised['start'] == 0 and normalised['stop'] == 1 and normalised['payload'] is native
    assert 'op' not in normalised
    done = Complete(tap_count=0, source_start=0, source_stop=0)
    finish = MailboxLink(Publisher(), Forward([done])).peek()
    assert finish['op'] == 'complete' and finish['payload'] is done


def test_an_unknown_message_kind_is_refused_rather_than_guessed():
    link = MailboxLink(Publisher(), Forward([object()]))
    with pytest.raises(LiveExchangeError):
        link.peek()


def test_a_completion_measures_nothing_and_commit_refuses_an_empty_preparation():
    sink = ForeignRowSink(None, None, (0,), Reader(), Bank(), 4, lambda: 0)
    done = MailboxLink(Publisher(), Forward([Complete(tap_count=0, source_start=0, source_stop=0)])).peek()
    assert sink.prepare(done) == (0, None)
    with pytest.raises(LiveExchangeError):
        sink.commit(None)
