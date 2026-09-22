"""Supply native GLM restoration with MCDMA staging and separate cache receipts."""
import hashlib
import math
import re
import time

from drift.exchange.lifetime import request_scope
from drift.serving.glm_restore_route import canonical_file
from drift.serving.live_publication import load_publication


class McdmaRestorationFactory:
    def __init__(self, publisher, *, layouts, max_rows):
        if (type(layouts) is not dict or not layouts or len(layouts) > 256
                or any(type(key) is not str or not re.fullmatch(r'l[0-9]+', key) or tuple(value) != (512,)
                       for key, value in layouts.items())
                or type(max_rows) is not int or not 1 <= max_rows <= 4096):
            raise ValueError('MCDMA_RESTORE_LAYOUT')
        self.publisher, self.layouts, self.max_rows = publisher, {key: (512,) for key in layouts}, max_rows

    def __call__(self, source, digest, session, remaining):
        remaining()
        path = canonical_file(str(source), 1048576)
        with path.open('rb') as stream:
            body = stream.read(1048577)
        if len(body) > 1048576:
            raise ValueError('MCDMA_RESTORE_SIZE')
        if hashlib.sha256(body).hexdigest() != digest:
            raise ValueError('MCDMA_RESTORE_DIGEST')
        arrays = load_publication(path, self.layouts, max_rows=self.max_rows)
        with path.open('rb') as stream:
            unchanged = stream.read(1048577) == body
        if not unchanged:
            raise ValueError('MCDMA_RESTORE_CHANGED')
        rows = next(iter(arrays.values())).shape[0]
        return RestorationPublication(self.publisher, session, body, digest, rows, remaining)


class RestorationPublication:
    def __init__(self, publisher, session, body, digest, rows, remaining):
        self.publisher, self.session, self.body, self.digest = publisher, session, body, digest
        self.rows, self.remaining, self.staged, self.delivered = rows, remaining, None, None
        self.delivery, self.failed, self.admitted = self, False, False

    def _check(self, deadline=None):
        try:
            remaining = self.remaining()
            valid = type(remaining) in (int, float) and math.isfinite(remaining) and remaining > 0
            valid = valid and (deadline is None or type(deadline) in (int, float) and math.isfinite(deadline) and time.monotonic() < deadline)
            if self.failed or not valid:
                raise TimeoutError('MCDMA_RESTORE_UNAVAILABLE')
        except Exception:
            self.failed = True
            raise

    def _deadline(self, timeout):
        self._check()
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            self.failed = True
            raise TimeoutError('MCDMA_RESTORE_UNAVAILABLE')
        return time.monotonic() + min(timeout, self.remaining())

    def _call(self, deadline, function, *args):
        try:
            with request_scope(lambda: self._check(deadline)):
                self._check(deadline)
                result = function(*args)
                self._check(deadline)
                return result
        except Exception:
            self.failed = True
            raise

    def health(self, *, deadline=None):
        self._check(deadline)
        if self.publisher.staging.failed:
            self.failed = True
            raise ValueError('MCDMA_RESTORE_TRANSPORT')

    def stage(self, *, timeout):
        if self.staged is not None or self.delivered is not None:
            self.failed = True
            raise ValueError('MCDMA_RESTORE_REUSE')
        self.staged = self._call(self._deadline(timeout),
                                 self.publisher.stage, self.session, 0, self.body, 1)

    def admit(self, sequence, digest, *, deadline=None):
        self._check(deadline)
        if type(sequence) is not int or sequence != 0 or digest != self.digest or self.staged is None:
            self.failed = True
            raise ValueError('MCDMA_RESTORE_ADMISSION')
        self.admitted = True

    def publish(self, *, timeout):
        if not self.admitted or self.delivered is not None:
            self.failed = True
            raise ValueError('MCDMA_RESTORE_RELEASE')
        self.delivered = self._call(self._deadline(timeout), self.publisher.release, self.staged)

    def wait_applied(self, sequence, rows, digest, *, deadline=None):
        self._check(deadline)
        if self.delivered is None or type(sequence) is not int or sequence != 0 or type(rows) is not int or rows != self.rows or digest != self.digest:
            self.failed = True
            raise ValueError('MCDMA_RESTORE_RECEIPT')
        result = self._call(deadline, self.publisher.confirm_applied, self.delivered, rows)
        try:
            receipts, ranks = result['receipts'], tuple(self.publisher.conns)
            if type(receipts) is not dict or set(receipts) != set(ranks):
                raise ValueError('MCDMA_RESTORE_RECEIPT')
            normalized = []
            for index, rank in enumerate(ranks):
                expected = dict(rank=index, world_size=len(ranks), sequence=0, rows=rows, sha256=digest)
                receipt = receipts[rank]
                if (type(receipt) is not dict or any(receipt.get(key) != value for key, value in expected.items())
                        or any(type(receipt.get(key)) is not int for key in ('rank', 'world_size', 'sequence', 'rows'))):
                    raise ValueError('MCDMA_RESTORE_RECEIPT')
                normalized.append(expected)
            return normalized
        except Exception:
            self.failed = True
            raise ValueError('MCDMA_RESTORE_RECEIPT') from None
