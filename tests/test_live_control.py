"""Exercise bounded delivery and cancellation without hosts or model data."""
import subprocess
import threading
from types import SimpleNamespace

import pytest

from drift.serving.live_control import PendingMemory, PendingMemoryFull, interrupt_processes


def test_upload_and_receiver_ack_both_hold_capacity():
    pending = PendingMemory(1)
    entered, release = threading.Event(), threading.Event()
    uploaded = []

    def upload():
        entered.set()
        assert release.wait(2)
        uploaded.append(True)

    producer = threading.Thread(target=lambda: pending.publish("private-path", 4, upload))
    producer.start()
    try:
        assert entered.wait(2)
        with pytest.raises(PendingMemoryFull):
            pending.publish("overflow", 2, lambda: pytest.fail("overflow reached transport"))
        assert not pending
        release.set()
        producer.join(2)
        assert not producer.is_alive()
        delivery = pending.take()
        assert delivery.rows == 4 and uploaded
        with pytest.raises(PendingMemoryFull):
            pending.publish("overflow", 2, lambda: pytest.fail("unacknowledged append released capacity"))
        assert pending.report() == {"capacity": 1, "pending": 1, "peak": 1, "pending_rows": 4}
        assert "private-path" not in str(pending.report())
        pending.complete(delivery)
        with pytest.raises(ValueError):
            pending.complete(delivery)
        pending.publish("next", 3, lambda: None)
        assert pending.take().path == "next"
    finally:
        release.set()
        producer.join(2)


def test_failed_upload_releases_capacity_without_publishing():
    pending = PendingMemory(1)

    def broken():
        raise OSError("synthetic upload failure")

    with pytest.raises(OSError):
        pending.publish("broken", 4, broken)
    assert not pending and pending.report()["pending"] == 0
    pending.publish("next", 2, lambda: None)
    assert pending.take().path == "next"


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        PendingMemory(capacity)


def test_interrupt_signals_every_process_before_waiting_then_escalates():
    calls = []

    def process(name, blocks=False):
        def wait(timeout):
            calls.append((name, "wait", timeout))
            if blocks:
                raise subprocess.TimeoutExpired(name, timeout)
        return SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append((name, "terminate")),
                               wait=wait, kill=lambda: calls.append((name, "kill")))

    interrupt_processes([process("a", True), process("b")])
    assert calls[:2] == [("a", "terminate"), ("b", "terminate")]
    assert ("a", "kill") in calls
    assert ("b", "wait", 2) in calls


def test_supervised_process_gets_abort_and_cleanup_time_before_escalation():
    calls = []
    pipe = SimpleNamespace(write=lambda value: calls.append(("write", value)),
                           flush=lambda: calls.append(("flush",)), close=lambda: calls.append(("close",)))
    remote = SimpleNamespace(stdin=pipe, poll=lambda: None,
                             terminate=lambda: calls.append(("remote", "terminate")),
                             wait=lambda **kw: calls.append(("remote", "wait")),
                             kill=lambda: calls.append(("remote", "kill")))
    other = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append(("other", "terminate")),
                            wait=lambda **kw: calls.append(("other", "wait")))
    interrupt_processes([remote, other], graceful=[remote])
    assert calls[:3] == [("write", '{"op":"abort"}\n'), ("flush",), ("close",)]
    assert calls.index(("other", "terminate")) < calls.index(("remote", "wait"))
    assert ("remote", "terminate") not in calls and ("remote", "kill") not in calls
