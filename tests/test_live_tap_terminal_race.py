"""Late asynchronous worker metadata cannot mutate an already completed stream."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace as NS

import numpy as np

from test_live_tap_tail import finished_request
from test_vllm_glm53_inject import connector


def test_late_worker_cannot_overwrite_scheduler_final_tail(connector):
    owner, root = connector
    owner._live['request'] = dict(name='terminal-race', failed='', next_seq=0, start=0, reserve=0, tap=True)
    shared = dict(name='terminal-race', request_id='request', reserve=0, reserve_start=0,
                  blocks=((0,), (0, 1, 2)), apply=(), tap=True, failed='')
    owner._live_step(NS(**shared, before=8, after=9, verified=8))
    owner._live_step(NS(**shared, before=19, after=20, verified=15))
    owner.request_finished(finished_request(20), ())
    folder = root / 'tp-live-out' / 'terminal-race'
    assert json.loads((folder / 'finished').read_text())['source_stop'] == 20
    before = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.glob('*.npz')}
    owner._live_step(NS(**shared, before=20, after=24, verified=16))
    after = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.glob('*.npz')}
    assert after == before
    with np.load(folder / '000001.npz') as arrays:
        assert int(arrays['stop']) == 20


def test_finalizer_holds_capture_lock_through_finished_marker(connector, monkeypatch):
    import drift.serving.live_tap_finish as finalizer
    owner, root = connector
    owner._live['request'] = dict(name='terminal-lock', failed='', next_seq=0, start=0, reserve=0, tap=True)
    shared = dict(name='terminal-lock', request_id='request', reserve=0, reserve_start=0,
                  blocks=((0,), (0, 1, 2)), apply=(), tap=True, failed='')
    owner._live_step(NS(**shared, before=8, after=9, verified=8))
    owner._live_step(NS(**shared, before=19, after=20, verified=15))
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    original = finalizer.finish_taps
    def finish(*args):
        result = original(*args)
        entered.set()
        assert release.wait(3)
        return result
    def capture():
        started.set()
        owner._live_step(NS(**shared, before=20, after=24, verified=16))
    monkeypatch.setattr(finalizer, 'finish_taps', finish)
    with ThreadPoolExecutor(max_workers=2) as executor:
        final = executor.submit(owner.request_finished, finished_request(20), ())
        assert entered.wait(3)
        late = executor.submit(capture)
        try:
            assert started.wait(3)
            assert not late.done()
        finally:
            release.set()
        final.result(timeout=3); late.result(timeout=3)
    folder = root / 'tp-live-out' / 'terminal-lock'
    with np.load(folder / '000001.npz') as arrays:
        assert int(arrays['stop']) == 20
