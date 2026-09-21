"""Exercise the live adapters against fake publishers, mailboxes and caches: no MLX, MCDMA or model."""
import numpy as np
import pytest

from drift.exchange.live import ForeignRowSink, LiveExchangeError, MailboxLink, OwnRowSource
from drift.serving.mcdma_forward import Complete, Tap


class Publisher:
    def __init__(self, receipts=None):
        self.receipts = receipts if receipts is not None else {
            'spark-b.invalid': {'sha256': 'b' * 64}, 'spark-a.invalid': {'sha256': 'b' * 64}}
        self.delivered = []

    def deliver(self, session, sequence, body, copies):
        self.delivered.append((session, sequence, len(body), copies))
        return dict(session=session, sequence=sequence, sha256='a' * 64)

    def confirm_applied(self, delivered, rows):
        return dict(receipts=self.receipts, applied_all_ranks_s=0.5)


class Forward:
    def __init__(self, queue=()):
        self.queue, self.acked = list(queue), []

    def peek(self):
        return self.queue.pop(0) if self.queue else None

    def acknowledge(self, tap):
        self.acked.append(tap)


def test_confirm_applied_reports_ranks_in_a_stable_order_with_their_digests():
    link = MailboxLink(Publisher(), Forward())
    applied = link.confirm_applied(dict(session='s', sequence=0, sha256='a' * 64), 12)
    assert applied['ranks'] == ('spark-a.invalid', 'spark-b.invalid')
    assert applied['receipts'] == ('b' * 64, 'b' * 64)


def test_the_link_passes_publication_arguments_through_unchanged():
    publisher = Publisher()
    MailboxLink(publisher, Forward()).deliver('live-1', 3, b'body', 12)
    assert publisher.delivered == [('live-1', 3, 4, 12)]


def test_a_publisher_failure_is_not_absorbed_by_the_link():
    publisher = Publisher()
    def fail(*_args, **_kwargs):
        raise RuntimeError('rank staging failed')
    publisher.confirm_applied = fail
    with pytest.raises(RuntimeError):
        MailboxLink(publisher, Forward()).confirm_applied({}, 1)


class Reader:
    def read(self, flat, gain=1.0):
        return {layer: np.asarray(value, dtype=np.float32) for layer, value in flat.items()}


def cache_for(rows, width=4):
    class Slot:
        def __init__(self):
            self.keys = np.zeros((1, 1, rows, width), dtype=np.float32)
            self.values = np.zeros((1, 1, rows, width), dtype=np.float32)
            self.offset = rows
    return [Slot() for _ in range(2)]


def test_own_row_source_packs_one_publication_per_layer_and_hashes_it(monkeypatch):
    import drift.exchange.live as live
    monkeypatch.setattr(live, 'tap_slots', lambda cache, layers, rope, slots, positions: {
        layer: (np.ones((len(slots), 2), dtype=np.float32), np.ones((len(slots), 2), dtype=np.float32)) for layer in layers})
    source = OwnRowSource(cache_for(3), None, (0, 1), (0, 1), Reader(), lambda: [0, 1, 2], lambda: [0, 1, 2])
    snapshot = source.snapshot(12)
    assert snapshot['rows'] == 3 and len(snapshot['sha256']) == 64 and snapshot['body'][:2] == b'PK'


def test_own_row_source_refuses_to_publish_nothing():
    source = OwnRowSource(cache_for(0), None, (0,), (0,), Reader(), lambda: [], lambda: [])
    with pytest.raises(LiveExchangeError):
        source.snapshot(12)


class ForeignReader:
    def read(self, latents):
        rows = len(next(iter(latents.values())))
        entries = {0: (np.ones((rows, 2), dtype=np.float32), np.ones((rows, 2), dtype=np.float32))}
        return entries, None, list(range(rows)), None


class Bank:
    def __init__(self):
        self.remembered = []

    def positions(self, sources, own):
        return np.asarray(sources, dtype=np.int64)

    def remember(self, entries, sources, first_slot):
        self.remembered.append(first_slot)


def test_prepare_measures_without_touching_the_cache(monkeypatch):
    import drift.exchange.live as live
    appended = []
    monkeypatch.setattr(live, 'append_entries', lambda *a, **k: appended.append(a) or 0)
    monkeypatch.setattr(live, 'validate_translation', lambda *a, **k: 2)
    sink = ForeignRowSink(cache_for(2), None, (0,), ForeignReader(), Bank(), 4, lambda: 10)
    rows, prepared = sink.prepare(Tap(meta={}, latents={0: np.ones((2, 2))}, start=0, stop=2))
    assert rows == 2 and appended == []
    assert sink.commit(prepared) == 2 and len(appended) == 1


def test_a_completion_marker_carries_no_rows_and_appends_nothing(monkeypatch):
    import drift.exchange.live as live
    monkeypatch.setattr(live, 'append_entries', lambda *a, **k: pytest.fail('completion must not append'))
    sink = ForeignRowSink(cache_for(0), None, (0,), ForeignReader(), Bank(), 4, lambda: 0)
    rows, prepared = sink.prepare(Complete(tap_count=1, source_start=0, source_stop=2))
    assert rows == 0 and prepared is None and sink.commit(prepared) == 0
