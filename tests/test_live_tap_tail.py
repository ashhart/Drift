"""Final tap coverage through the real connector and bridge, using synthetic caches."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_vllm_glm53_inject import connector
from test_mcdma_forward import Region
from drift.serving.mcdma_mailbox import Reader
from scripts.mcdma_target.spark_bridge import Bridge


def finished_request(stop=12, **changes):
    fields = dict(request_id='request', num_computed_tokens=stop, num_tokens=stop + 1,
                  num_in_flight_tokens=0, num_stale_output_tokens=0,
                  status=NS(name='FINISHED_STOPPED'))
    return NS(**(fields | changes))


def run_steps(c):
    c._live['request'] = dict(name='tail-test', failed='', next_seq=0, start=0, reserve=0, tap=True)
    shared = dict(name='tail-test', request_id='request', reserve=0, reserve_start=0,
                  blocks=((0,), (0, 1)), apply=(), tap=True, failed='')
    c._live_step(NS(**shared, before=8, after=9, verified=8))
    c._live_step(NS(**shared, before=11, after=12, verified=11))


def test_terminal_covers_last_computed_rows(connector):
    c, root = connector
    run_steps(c)
    c.request_finished(finished_request(), ())
    folder = root / 'tp-live-out' / 'tail-test'
    region = Region()
    reader = Reader(region, session=9)
    bridge = Bridge(None, None, str(root / 'in'), str(root / 'tp-live-out'), 0, True, region)
    bridge.watching = 'tail-test'
    exported = []
    for _ in range(4):
        bridge.publish_taps()
        payload = reader.peek()
        assert payload is not None
        meta = json.loads(payload.partition(b'\n')[0])
        if meta.get('op') == 'complete':
            break
        with np.load(folder / f"{meta['tap']:06d}.npz") as values:
            exported.append((int(values['start']), int(values['stop'])))
        reader.acknowledge()
    assert exported[-1][1] == 12
    assert exported == [(0, 8), (8, 12)]
    assert meta == dict(op='complete', session='tail-test', tap_count=2, bytes=0, source_start=0, source_stop=12)


@pytest.mark.parametrize('computed,tokens,stop', [(11, 13, 11), (12, 11, 11)])
def test_speculative_or_stop_trimmed_rows_are_never_published(connector, computed, tokens, stop):
    c, root = connector
    run_steps(c)
    c.request_finished(finished_request(computed, num_tokens=tokens), ())
    folder = root / 'tp-live-out' / 'tail-test'
    with np.load(folder / '000001.npz') as tail:
        assert (int(tail['start']), int(tail['stop'])) == (8, stop)
        assert tail['l3'].shape == (stop - 8, 512)
    assert json.loads((folder / 'finished').read_text())['source_stop'] == stop


def test_finalizer_never_rereads_freed_or_reused_device_pages(connector):
    c, root = connector
    run_steps(c)
    c._kv_caches.clear()
    c._live_worker.clear()
    c.request_finished_all_groups(finished_request(), ())
    folder = root / 'tp-live-out' / 'tail-test'
    assert json.loads((folder / 'finished').read_text())['source_stop'] == 12
    assert (folder / '000001.npz').exists()


@pytest.mark.parametrize('changes', [dict(num_in_flight_tokens=2), dict(num_stale_output_tokens=1),
                                   dict(num_computed_tokens=None), dict(num_computed_tokens=True),
                                   dict(num_computed_tokens=13), dict(num_computed_tokens=10),
                                   dict(status=NS(name='FINISHED_ABORTED')),
                                   dict(status=NS(name='FINISHED_ERROR'))])
def test_unsettled_or_invalid_frontier_cannot_complete(connector, changes):
    c, root = connector
    run_steps(c)
    c.request_finished(finished_request(**changes), ())
    folder = root / 'tp-live-out' / 'tail-test'
    assert json.loads((folder / 'finished').read_text())['failed'] == 'live tap terminal coverage failed'
    assert not (folder / '000001.npz').exists()


@pytest.mark.parametrize('fault', ['missing', 'corrupt', 'worker_error', 'gap'])
def test_missing_corrupt_or_failed_snapshot_cannot_complete(connector, fault):
    c, root = connector
    run_steps(c)
    folder = root / 'tp-live-out' / 'tail-test'
    pending = folder / '.pending' / 'tail.npz'
    if fault == 'missing':
        pending.unlink()
    elif fault == 'corrupt':
        pending.write_bytes(b'synthetic corruption')
    elif fault == 'worker_error':
        (folder / 'error.rank1').write_text('synthetic device failure')
    else:
        (folder / '000000.npz').rename(folder / '000002.npz')
    c.request_finished(finished_request(), ())
    assert json.loads((folder / 'finished').read_text())['failed']
    assert not (folder / '000001.npz').exists()


def test_pending_candidate_is_never_a_forward_publication(connector):
    c, root = connector
    run_steps(c)
    folder = root / 'tp-live-out' / 'tail-test'
    assert [path.name for path in folder.glob('*.npz')] == ['000000.npz']
    assert (folder / '.pending' / 'tail.npz').exists()
    region = Region()
    reader = Reader(region, session=9)
    bridge = Bridge(None, None, str(root / 'in'), str(root / 'tp-live-out'), 0, True, region)
    bridge.watching = 'tail-test'
    bridge.publish_taps()
    reader.acknowledge()
    bridge.publish_taps()
    assert reader.peek() is None
    assert not bridge.forward_complete


def test_nonhead_rank_does_not_stage_outbound_candidates(connector):
    c, root = connector
    c._tp = lambda: (1, 2)
    run_steps(c)
    assert not (root / 'tp-live-out' / 'tail-test').exists()


def test_short_first_step_flushes_only_own_span_at_finish(connector):
    c, root = connector
    c._live['request'] = dict(name='short-test', failed='', next_seq=0, start=2, reserve=3, tap=True)
    c._live_step(NS(name='short-test', request_id='request', reserve=3, reserve_start=2,
                    blocks=((0,), (0, 1)), apply=(), tap=True, before=0, after=7, verified=0))
    c.request_finished(finished_request(7, status=NS(name='FINISHED_LENGTH_CAPPED')), ())
    folder = root / 'tp-live-out' / 'short-test'
    with np.load(folder / '000000.npz') as tail:
        assert (int(tail['start']), int(tail['stop'])) == (5, 7)
        assert tail['l3'].shape == (2, 512)


def test_finished_callback_is_idempotent(connector):
    c, root = connector
    run_steps(c)
    request = finished_request()
    c.request_finished(request, ())
    folder = root / 'tp-live-out' / 'tail-test'
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    c.request_finished_all_groups(request, ())
    assert before == {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}


def test_final_file_appearing_during_bridge_poll_retries_instead_of_false_gap(connector, monkeypatch):
    import scripts.mcdma_target.spark_bridge as module
    c, root = connector
    run_steps(c)
    c.request_finished(finished_request(), ())
    region = Region()
    reader = Reader(region, session=9)
    bridge = Bridge(None, None, str(root / 'in'), str(root / 'tp-live-out'), 0, True, region)
    bridge.watching = 'tail-test'
    original = module.os.path.exists
    monkeypatch.setattr(module.os.path, 'exists', lambda path: False if str(path).endswith('000000.npz') else original(path))
    bridge.publish_taps()
    assert reader.peek() is None
    assert not bridge.forward_complete
    monkeypatch.setattr(module.os.path, 'exists', original)
    bridge.publish_taps()
    assert json.loads(reader.peek().partition(b'\n')[0])['tap'] == 0
