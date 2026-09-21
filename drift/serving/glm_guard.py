"""Admit an idle owned engine and observe bounded post-cancellation recovery."""
import time


class EngineGuard:
    def __init__(self, sample, *, clock=time.monotonic, pause=time.sleep):
        self.sample, self.clock, self.pause = sample, clock, pause
        self.baseline, self.deadline = None, None

    def _remaining(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('engine recovery deadline')
        return remaining

    def admit(self, timeout):
        self.deadline = self.clock() + timeout
        state = self.sample(min(5, self._remaining()))
        if state.running != 0 or state.waiting != 0:
            raise ValueError('engine is occupied')
        self._remaining()
        self.baseline = state

    def recover(self):
        if self.baseline is None:
            return
        settled = 0
        while settled < 3:
            state = self.sample(min(5, self._remaining()))
            self._remaining()
            if state.running > 1 or state.waiting:
                raise ValueError('concurrent engine work makes recovery ambiguous')
            idle = state.running == 0 and state.kv_usage <= self.baseline.kv_usage
            settled = settled + 1 if idle else 0
            if settled < 3:
                self.pause(min(.2, self._remaining()))

