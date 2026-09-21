"""Sequence one route's bounded exchanges, enforcing mode, cursors and fail-closed poisoning."""
from drift.exchange.contract import MODES, validate_exchange
from drift.exchange import validation
from drift.exchange.lifetime import checkpoint

MAX_SEQUENCE = 1_000_000
DEFAULT_ROW_CAP = 2048


class ExchangeError(RuntimeError):
    pass


class ExchangeSession:
    """Owns the cursors a live route cannot re-derive; the runtime is injected as source, link and sink."""

    def __init__(self, *, session, source_worker, target_worker, mode, ranks, source, link, sink,
                 copies=1, row_cap=DEFAULT_ROW_CAP, publish_row_cap=DEFAULT_ROW_CAP):
        if mode not in MODES:
            raise ExchangeError('EXCHANGE_MODE')
        if source_worker == target_worker:
            raise ExchangeError('EXCHANGE_ROUTE')
        self.session, self.source_worker, self.target_worker = session, source_worker, target_worker
        self.mode, self.ranks = mode, tuple(ranks) if mode == 'drift' else ()
        try:
            for value in (session, source_worker, target_worker):
                validation.identity(value)
            validation.integer(copies, 1, 64)
            validation.integer(row_cap, 1, validation.MAX_ROWS)
            validation.integer(publish_row_cap, 1, validation.MAX_ROWS)
            if mode == 'drift':
                validation.ranks(self.ranks)
        except ValueError as error:
            raise ExchangeError(str(error)) from error
        self.source, self.link, self.sink = source, link, sink
        self.copies, self.row_cap = copies, row_cap
        self.publish_row_cap, self.published_rows, self.source_stop = publish_row_cap, 0, None
        self.sequence, self.foreign_rows, self.poisoned = 0, 0, False
        self.source_start, self.tap_count, self.forward_complete = None, 0, False

    def _poison(self, code, cause=None):
        """Poison first, then raise; an ExchangeError from a collaborator keeps its own code."""
        self.poisoned = True
        if isinstance(cause, ExchangeError) and cause.args:
            raise ExchangeError(cause.args[0]) from cause
        raise ExchangeError(code) from cause

    def _guard(self):
        if self.poisoned:
            raise ExchangeError('EXCHANGE_POISONED')
        self._checkpoint()
        if self.sequence >= MAX_SEQUENCE:
            self._poison('EXCHANGE_SEQUENCE')

    def _checkpoint(self):
        try:
            checkpoint()
        except Exception as error:
            self._poison('EXCHANGE_CANCELLED', error)

    def _invoke(self, code, call, *args):
        self._checkpoint()
        try:
            result = call(*args)
        except Exception as error:
            self._poison(code, error)
        self._checkpoint()
        return result

    def publish_own(self, text_bytes=0):
        """Publish this side's own rows, or record the text arm; the returned record is already validated."""
        self._guard()
        if self.mode == 'drift' and text_bytes:
            self._poison('EXCHANGE_TEXT_FALLBACK')
        try:
            validation.integer(text_bytes, 0, validation.MAX_BYTES)
            record = self._text_record(text_bytes) if self.mode == 'text' else self._drift_record()
            validate_exchange(record, session=self.session, sequence=self.sequence, mode=self.mode,
                              source_worker=self.source_worker, target_worker=self.target_worker,
                              expected_ranks=self.ranks)
        except ValueError as error:
            self._poison(str(error))
        except Exception as error:
            self._poison('EXCHANGE_RESULT_FAILED', error)
        self._checkpoint()
        self.sequence += 1
        self.published_rows += record['rows']
        return record

    def _text_record(self, text_bytes):
        return self._record(rows=0, source_rows=0, copies=0, size=0, sha256=None,
                            text_bytes=text_bytes, applied=(), receipts=())

    def _drift_record(self):
        try:
            snapshot = self._invoke('EXCHANGE_LINK_FAILED', self.source.snapshot, self.copies)
            rows = validation.snapshot(snapshot, self.copies)
            if self.published_rows + rows > self.publish_row_cap:
                self._poison('EXCHANGE_PUBLISH_CAP')
            delivered = self._invoke('EXCHANGE_LINK_FAILED', self.link.deliver,
                                     self.session, self.sequence, snapshot['body'], self.copies)
            applied = self._invoke('EXCHANGE_LINK_FAILED', self.link.confirm_applied, delivered, rows)
            return self._record(rows=rows, source_rows=snapshot['rows'], copies=self.copies,
                                size=len(snapshot['body']), sha256=snapshot['sha256'], text_bytes=0,
                                applied=tuple(applied['ranks']), receipts=tuple(applied['receipts']))
        except Exception as error:
            self._poison('EXCHANGE_LINK_FAILED', error)

    def _record(self, *, rows, source_rows, copies, size, sha256, text_bytes, applied, receipts):
        return dict(v=1, session=self.session, sequence=self.sequence, mode=self.mode,
                    source_worker=self.source_worker, target_worker=self.target_worker,
                    rows=rows, source_rows=source_rows, copies=copies, bytes=size, source_sha256=sha256,
                    text_bytes=text_bytes, applied_ranks=applied, receipts=receipts)

    def drain_forward(self):
        """Apply every committed partner tap in order; the text arm never reads the link at all."""
        self._guard()
        if self.mode != 'drift':
            return []
        applied = []
        while True:
            tap = self._invoke('EXCHANGE_LINK_FAILED', self.link.peek)
            if tap is None:
                return applied
            if self.forward_complete:
                self._poison('EXCHANGE_COMPLETE')
            if type(tap) is dict and tap.get('op') == 'complete':
                self._complete(tap)
                return applied
            try:
                start, stop = validation.frontier(tap, self.source_stop)
            except Exception as error:
                self._poison('EXCHANGE_TAP', error)
            rows, prepared = self._prepare(tap)
            if self.foreign_rows + rows > self.row_cap:
                self._poison('EXCHANGE_FOREIGN_CAP')
            self._commit(prepared, rows)
            self.foreign_rows += rows
            self._invoke('EXCHANGE_ACK_FAILED', self.link.acknowledge, tap)
            self.source_stop = stop
            if self.source_start is None:
                self.source_start = start
            self.tap_count += 1
            applied.append(dict(session=self.session, rows=rows, start=start, stop=stop))

    def _complete(self, tap):
        try:
            validation.completion(tap, self.tap_count, self.source_start, self.source_stop)
        except Exception as error:
            self._poison('EXCHANGE_COMPLETE', error)
        self._invoke('EXCHANGE_ACK_FAILED', self.link.acknowledge, tap)
        self.forward_complete = True

    def _prepare(self, tap):
        """Translate and validate without touching the cache, so the row cap refuses before any append."""
        try:
            rows, prepared = self._invoke('EXCHANGE_PREPARE_FAILED', self.sink.prepare, tap)
        except Exception as error:
            self._poison('EXCHANGE_PREPARE_FAILED', error)
        if type(rows) is not int or not 1 <= rows <= validation.MAX_ROWS:
            self._poison('EXCHANGE_APPLY_ROWS')
        return rows, prepared

    def _commit(self, prepared, rows):
        """Append the already measured rows; a count that disagrees with the measurement poisons the session."""
        added = self._invoke('EXCHANGE_APPLY_FAILED', self.sink.commit, prepared)
        if type(added) is not int or added != rows:
            self._poison('EXCHANGE_APPLY_ROWS')
