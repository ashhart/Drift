"""Own one all-rank staging transaction until its head publication is released."""
from contextvars import copy_context
import hashlib
import re
import threading
import time

from drift.exchange.lifetime import checkpoint
from drift.serving.mcdma_mailbox import MailboxError


class Staging:
    def __init__(self, publisher, envelope):
        self.publisher, self.envelope = publisher, envelope
        self.lock, self.pending, self.failed = threading.RLock(), None, False

    def stage(self, session, sequence, body, copies):
        with self.lock:
            # Refused before anything is sent: rejecting a caller must not disable the transport.
            if self.failed or self.pending is not None:
                raise MailboxError('staging transaction unavailable')
            try:
                if (type(session) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', session)
                        or type(sequence) is not int or not 0 <= sequence < 10**6
                        or type(copies) is not int or not 1 <= copies <= 64 or type(body) is not bytes or not body):
                    raise MailboxError('invalid staging publication')
                checkpoint()
                started, errors = time.perf_counter(), {}

                def send(rank):
                    try:
                        self.publisher._deliver(rank, self.envelope('stage', session, sequence, body, copies))
                    except Exception as error:
                        errors[rank] = error

                threads = [threading.Thread(target=copy_context().run, args=(send, rank)) for rank in self.publisher.conns]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if errors:
                    raise MailboxError('staging failed; nothing was released')
                checkpoint()
                record = dict(session=session, sequence=sequence, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                              staged_all_ranks_s=round(time.perf_counter() - started, 6))
                self.pending = record, dict(record), started
                return record
            except Exception:
                self.failed = True
                raise

    def release(self, record):
        with self.lock:
            if self.failed or self.pending is None or record is not self.pending[0] or record != self.pending[1]:
                raise MailboxError('staging release does not match its owner')
            try:
                checkpoint()
                self.publisher._deliver(self.publisher.head, self.envelope('release', record['session'], record['sequence']))
                checkpoint()
                result = {**record, 'released_s': round(time.perf_counter() - self.pending[2], 6), '_t0': time.perf_counter()}
                self.pending = None
                return result
            except Exception:
                self.failed = True
                raise
