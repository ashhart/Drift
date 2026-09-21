"""Share owned subprocess mechanics without choosing linked or no-link command authority."""
import json
import os
import selectors
import subprocess
import time

from drift.serving.live_qualification_events import EventCounts, QualificationError, RequestEvidence


class OwnedProcess:
    def __init__(self, command, claim, limits, reserve, counts=None):
        self.command, self.claim, self.limits = list(command), claim, limits
        self.reserve = reserve
        self.counts = counts if counts is not None else EventCounts(limits, reserve)
        self.child = self.selector = None
        self.buffer = bytearray()
        self.abort_sent = self.stdout_closed = self.escalated = False
        self.failure = self.started_at = None
        self.closed = False

    def start(self):
        if self.started_at is not None:
            raise QualificationError('SESSION_REUSE')
        self.claim.spend()
        self.started_at = time.monotonic()
        try:
            self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.child.stdout, selectors.EVENT_READ)
        except Exception:
            self.failure = 'START_FAILURE'
            raise QualificationError(self.failure) from None

    def _reject(self, code):
        self.failure = self.failure or code
        self.close()
        raise QualificationError(code) from None

    def poll(self, timeout=0):
        if self.child is None:
            self._reject('NOT_STARTED')
        cleanup = min(2 * self.limits.stop_timeout + 0.25, self.limits.max_seconds / 2)
        remaining = self.limits.max_seconds - cleanup - (time.monotonic() - self.started_at)
        if remaining <= 0:
            self._reject('WALL_LIMIT')
        try:
            ready = self.selector.select(max(0, min(timeout, remaining))) if not self.stdout_closed else []
            for key, _ in ready:
                chunk = os.read(key.fd, min(4096, self.limits.max_line_bytes + 1))
                if not chunk:
                    self.stdout_closed = True
                    self.selector.unregister(key.fileobj)
                    if self.buffer:
                        self._reject('TRUNCATED_LINE')
                    continue
                self.buffer.extend(chunk)
                while b'\n' in self.buffer:
                    raw, _, rest = self.buffer.partition(b'\n')
                    self.buffer = bytearray(rest)
                    if len(raw) > self.limits.max_line_bytes:
                        self._reject('LINE_LIMIT')
                    self.counts.accept(json.loads(raw))
                if len(self.buffer) > self.limits.max_line_bytes:
                    self._reject('LINE_LIMIT')
            if self.counts.cancelled and not self.abort_sent:
                self._reject('UNSOLICITED_CANCELLATION')
            code = self.child.poll()
            if self.stdout_closed and code is not None:
                valid_done = code == 0 and self.counts.done and not self.counts.cancelled
                valid_cancel = code == 2 and self.counts.cancelled and self.abort_sent and not self.counts.done
                if not (valid_done or valid_cancel):
                    self._reject('CHILD_EXIT_PROTOCOL')
        except QualificationError as error:
            self._reject(str(error))
        except Exception:
            self._reject('EVENT_DECODE_FAILURE')
        if time.monotonic() - self.started_at >= self.limits.max_seconds - cleanup:
            self._reject('WALL_LIMIT')

    pump = poll

    def request_state(self):
        code = None if self.child is None else self.child.poll()
        active = bool(self.counts.nonempty and code is None and not self.counts.terminal and not self.abort_sent and not self.failure)
        cancelled = bool(self.abort_sent and self.counts.cancelled and not self.counts.done
                         and code == 2 and self.stdout_closed and not self.failure)
        completed = bool(self.counts.done or code == 0)
        return RequestEvidence(self.child is not None, active, self.abort_sent, cancelled, completed)

    def abort(self):
        self.poll(0)
        if not self.request_state().active:
            self._reject('NOT_ACTIVE_FOR_ABORT')
        try:
            self.child.stdin.write(b'{"op":"abort"}\n')
            self.child.stdin.flush()
        except (OSError, ValueError):
            self._reject('ABORT_SEND_FAILURE')
        self.abort_sent = True

    def close(self):
        if self.child is None or self.closed:
            return
        if self.child.poll() is None:
            self.escalated = True
            self.child.terminate()
        try:
            self.child.wait(timeout=2 * self.limits.stop_timeout + 0.25)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=self.limits.stop_timeout)
        if self.selector is not None:
            self.selector.close()
        self.child.stdin.close()
        self.child.stdout.close()
        self.buffer.clear()
        self.closed = True

    def report(self):
        code = None if self.child is None else self.child.poll()
        return {**self.counts.report(), 'session': self.claim.name, 'abort_sent': self.abort_sent,
                'cancelled': self.request_state().cancelled, 'completed': self.request_state().completed,
                'child_pid': None if self.child is None else self.child.pid, 'child_returncode': code,
                'child_terminated': self.child is not None and code is not None, 'stdout_closed': self.stdout_closed,
                'process_role': 'local_session_supervisor', 'termination_escalated': self.escalated,
                'failure': self.failure, 'cleanup_allowance_seconds': 2 * self.limits.stop_timeout + 0.25,
                'wall_seconds': None if self.started_at is None else time.monotonic() - self.started_at}
