"""Async scheduler counters must not certify unfinished speculative cache rows."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_vllm_glm53_inject import connector, live_request, step


def test_async_terminal_trims_to_processed_frontier(connector):
    owner, root = connector
    request = live_request('async', reserve=2, prompt=19, name='async-tail')
    request.num_in_flight_tokens = request.num_stale_output_tokens = 0
    owner.on_new_request(request)
    owner.meta = owner.build_connector_meta(step(new=[('async', ([0], [0, 1, 2, 3, 4, 5]), 0)], scheduled={'async': 19}))
    owner.wait_for_save()
    request.num_in_flight_tokens = 8
    for before, count in ((19, 8), (27, 7), (34, 8)):
        owner.meta = owner.build_connector_meta(step(cached=[('async', None, before)], scheduled={'async': count}))
        owner.wait_for_save()
    folder = root / 'tp-live-out' / 'async-tail'
    for path in folder.glob('*.npz'):
        with np.load(path) as arrays:
            assert int(arrays['stop']) <= 26
    request.num_computed_tokens, request.num_tokens = 39, 31
    request.status = NS(name='FINISHED_STOPPED')
    owner.request_finished(request, ())
    outcome = json.loads((folder / 'finished').read_text())
    assert outcome['failed'] == ''
    assert outcome['source_stop'] == 31
    cursor = 2
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as arrays:
            assert int(arrays['start']) == cursor
            cursor = int(arrays['stop'])
            assert cursor <= 31
    assert cursor == 31


@pytest.mark.parametrize('inflight,stale', [(None, 0), (True, 0), (-1, 0), (1, 0), (0, 1)])
def test_invalid_async_counters_refuse_before_cache_write(connector, inflight, stale):
    owner, _ = connector
    request = live_request('bad', 2, prompt=19)
    request.num_in_flight_tokens, request.num_stale_output_tokens = inflight, stale
    owner.on_new_request(request)
    owner.meta = owner.build_connector_meta(step(new=[('bad', ([0], [0, 1, 2]), 0)], scheduled={'bad': 19}))
    assert owner.meta.live[0].failed
    with pytest.raises(RuntimeError, match='poisoned'):
        owner.wait_for_save()
