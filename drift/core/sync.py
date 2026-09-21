from __future__ import annotations
from collections.abc import Callable
from .memory import ForeignKVBank, ForeignView
from .types import Delta


class SyncController:
    """Single-threaded, one-epoch-lag reference scheduler.

    A(k) and B(k) both pin their incoming state BEFORE either computes/publishes.
    Any error poisons this controller; resume requires a recorded checkpoint/replay.
    A bank's update is atomic across layers. A distributed pair is NOT a global
    transaction; failure after one publication aborts the run, never continues it.
    """
    def __init__(self, incoming_a: ForeignKVBank, incoming_b: ForeignKVBank):
        if incoming_a.direction != 1 or incoming_b.direction != 0:
            raise ValueError("incorrect incoming-bank direction wiring")
        self.a, self.b, self.epoch, self.failed = incoming_a, incoming_b, 0, False

    def tick(self, compute_a: Callable[[ForeignView | None, int], Delta],
             compute_b: Callable[[ForeignView | None, int], Delta]) -> None:
        if self.failed:
            raise RuntimeError("controller is poisoned; restore a clean run")
        try:
            a_view, b_view = self.a.pin(), self.b.pin()
            for view in (a_view, b_view):
                if view is not None and view.epoch != self.epoch - 1:
                    raise ValueError("lockstep requires precisely the prior epoch")
            a_delta = compute_a(a_view, self.epoch)
            b_delta = compute_b(b_view, self.epoch)
            if a_delta.epoch != self.epoch or b_delta.epoch != self.epoch:
                raise ValueError("worker returned a different epoch")
            self.b.commit(a_delta)
            self.a.commit(b_delta)
            self.epoch += 1
        except Exception:
            self.failed = True
            raise
