"""Wake ordinary own-input processing on owner failure or its absolute deadline."""
import os
import select
import time


class OwnerInput:
    def __init__(self, fd, deadline, failed):
        self.fd, self.deadline, self.failed = fd, deadline, failed
        self.pending = bytearray()
        self.reason = None

    def readline(self, maximum):
        while True:
            if self.failed(): self.reason = 'owner_control'; return ''
            remaining = self.deadline - time.monotonic()
            if remaining <= 0: self.reason = 'deadline'; return ''
            if b'\n' in self.pending:
                end = self.pending.index(b'\n') + 1
                raw = bytes(self.pending[:end]); del self.pending[:end]
                if len(raw) > maximum: raise ValueError('OWN_INPUT_LIMIT')
                return raw.decode('utf-8')
            if len(self.pending) >= maximum: raise ValueError('OWN_INPUT_LIMIT')
            if not select.select([self.fd], [], [], min(.05, remaining))[0]: continue
            raw = os.read(self.fd, min(4096, maximum - len(self.pending)))
            if not raw:
                self.reason = 'input_eof'
                if self.pending: raise ValueError('OWN_INPUT_TRUNCATED')
                return ''
            self.pending.extend(raw)
