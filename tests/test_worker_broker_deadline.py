import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from drift.serving.worker_broker_lifetime import BrokerWorker


def test_finish_never_waits_past_absolute_deadline_and_preserves_failure(tmp_path):
    clock = [10.0]
    class Process:
        stdin = None
        def poll(self): return None
        def communicate(self, timeout):
            assert timeout <= .25
            clock[0] += timeout
            raise __import__('subprocess').TimeoutExpired('synthetic', timeout)
        def terminate(self): pass
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe()
    worker = BrokerWorker(Process(), control_w, evidence_r, tmp_path, tmp_path/'config', 'pin', None, 10.25, 2)
    try:
        with patch('drift.serving.worker_broker_lifetime.time.monotonic', lambda: clock[0]):
            with pytest.raises(RuntimeError, match='BROKER_CLEANUP_UNCONFIRMED'):
                worker.finish(abort=True)
        assert tmp_path.exists()
    finally:
        os.close(control_r); os.close(evidence_w)


def test_missing_evidence_with_inherited_writer_cannot_block_cleanup(tmp_path):
    import time
    class Process:
        stdin = None
        def communicate(self, timeout): return b'', b''
    control_r, control_w = os.pipe(); evidence_r, evidence_w = os.pipe()
    started = time.monotonic()
    worker = BrokerWorker(Process(), control_w, evidence_r, tmp_path, tmp_path/'config', 'pin', None, started+.15, .05)
    try:
        with pytest.raises(RuntimeError, match='BROKER_CLEANUP_UNCONFIRMED'):
            worker.finish()
        assert time.monotonic()-started < .5 and tmp_path.exists()
    finally:
        os.close(control_r); os.close(evidence_w)
