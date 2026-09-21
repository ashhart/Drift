"""Admit and release one bounded publication, retaining verified rank receipts only."""
import math
import re
import time


class SinglePublication:
    """This delivery gate excludes model use, TP atomicity and worker-stop qualification."""

    def __init__(self, delivery, *, rows, digest, timeout, no_link=False, clock=time.monotonic):
        if type(rows) is not int or not 1 <= rows <= 12:
            raise ValueError('single-publication row limit exceeded')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('publication digest is invalid')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 180:
            raise ValueError('single-publication timeout is invalid')
        if type(no_link) is not bool or (not no_link and delivery is None):
            raise ValueError('linked publication requires rank delivery')
        self.delivery, self.rows, self.digest = delivery, rows, digest
        self.timeout, self.no_link, self.clock = timeout, no_link, clock
        self.used = False

    def run(self, stage, publish_and_release):
        """Callbacks receive remaining timeout; the caller owns request cleanup in every outcome."""
        if self.used:
            raise RuntimeError('single-publication gate cannot be reused')
        self.used = True
        started = self.clock()
        deadline = started + self.timeout
        result = {'status': 'INVALID', 'scope': 'single_publication_delivery', 'linked': not self.no_link,
                  'sequence': 0, 'rows': self.rows, 'sha256': self.digest, 'staged': 0,
                  'released': 0, 'receipts': [], 'phase': 'PREFLIGHT',
                  'worker_shutdown': 'BLOCKED', 'causal_use': 'NOT_TESTED', 'tp_atomicity': 'NOT_ESTABLISHED'}
        def remaining():
            value = deadline - self.clock()
            if value <= 0:
                raise TimeoutError('publication deadline exceeded')
            return value
        try:
            if self.no_link:
                result.update(status='PASSED', scope='no_link_transport', phase='CLOSED', rows=0, sha256=None)
            else:
                self.delivery.health(deadline=deadline)
                remaining()
                result['phase'] = 'STAGING'
                stage(timeout=remaining())
                remaining()
                result['staged'] = 1
                result['phase'] = 'ADMISSION'
                self.delivery.admit(0, self.digest, deadline=deadline)
                remaining()
                result['phase'] = 'RELEASE'
                publish_and_release(timeout=remaining())
                remaining()
                result['released'] = 1
                result['phase'] = 'RECEIPTS'
                result['receipts'] = self.delivery.wait_applied(0, self.rows, self.digest, deadline=deadline)
                remaining()
                self.delivery.health(deadline=deadline)
                remaining()
                result.update(status='PASSED', phase='DELIVERED')
        except Exception as error:
            result.update(status='INVALID', error_type=type(error).__name__)
        result['elapsed_seconds'] = max(0, self.clock() - started)
        return result
