"""Collect consecutive native requests through the real bridge and mailbox codec."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from drift.serving.mcdma_mailbox import Reader
from tests.test_mcdma_forward import Region
from tests.test_mcdma_mailbox import _bridge_module


def setup(tmp_path):
    from drift.serving.mcdma_turn_outbox import McdmaTurnOutbox
    root = tmp_path / 'exports'
    root.mkdir(mode=0o700)
    native = tmp_path / 'native'
    native.mkdir(mode=0o700)
    region = Region()
    reader = Reader(region, session=91)
    module = _bridge_module()
    bridge = module.Bridge(None, None, str(tmp_path / 'in'), str(native), 0, True, region)
    class Publisher:
        def watch(self, name):
            bridge.handle(module.pack('watch', name, 0))
    config = dict(memory_root=str(root), source_worker='glm', target_worker='qwen', recipe_sha256='a'*64,
                  max_raw_rows=32, max_raw_bytes=65536, max_rows=8, max_bytes=32768, max_publications=4)
    collector = McdmaTurnOutbox(config, {'l3': (512,)}, reader, Publisher(),
                               pause=lambda _: bridge.publish_taps())
    proof = dict(verified=True, prompt_tokens=12, reserve_start=3, reserve_tokens=2)
    return collector, bridge, native, proof


def output(native, name, *, start=5, stop=15):
    folder = native / name
    folder.mkdir(mode=0o700)
    np.savez(folder / '000000.npz', start=np.array(start), stop=np.array(stop),
             l3=np.ones((stop-start, 512), dtype=np.float16))
    (folder / 'finished').write_text(json.dumps(dict(failed='', writes_scheduled=1, tap_count=1,
                                                   source_start=start, source_stop=stop)))


def test_two_turns_keep_wire_sequence_and_ack_only_captured_rows(tmp_path):
    collector, bridge, native, proof = setup(tmp_path)
    for index in range(2):
        name = f'turn-{index}'
        collector.begin(name, proof, 8)
        output(native, name)
        report = collector.finish(name, 8, time.monotonic()+2)
        assert report['selected_rows'] == 3 and report['verified_stop'] == 15
        assert report['transport'] == 'mcdma' and report['cache_applied'] is False
        assert report['final_tail'] == 'UNKNOWN' and report['full_completion'] is False
        manifest = Path(report['manifest_path'])
        details = json.loads(manifest.read_bytes())
        assert details['publications'][0]['source_start'] == 12
        with np.load(manifest.parent / details['publications'][0]['file'], allow_pickle=False) as arrays:
            assert arrays['l3'].shape == (3, 512)
        assert bridge.forward.acknowledged()
    assert bridge.forward.sequence == 4 and collector.reader.expected == 5


@pytest.mark.parametrize('fault', ['foreign', 'gap', 'overrun', 'row_cap', 'bytes', 'publications'])
def test_rejected_rows_do_not_ack_or_complete(tmp_path, fault):
    collector, bridge, native, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    output(native, 'turn-1', start=4 if fault == 'foreign' else 6 if fault == 'gap' else 5,
           stop=21 if fault == 'overrun' else 15)
    if fault == 'row_cap': collector.rows = 2
    if fault == 'bytes': collector.raw_bytes = 1
    if fault == 'publications': collector.publications = 0
    bridge.publish_taps()
    with pytest.raises(ValueError): collector.finish('turn-1', 8, time.monotonic()+2)
    assert collector.failed and not bridge.forward.acknowledged()
    assert not list(collector.root.glob('*/manifest.json'))


def test_failed_capture_does_not_ack(tmp_path, monkeypatch):
    import drift.serving.mcdma_turn_outbox as module
    collector, bridge, native, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    output(native, 'turn-1')
    bridge.publish_taps()
    def fail(*args): raise OSError('synthetic full disk')
    monkeypatch.setattr(module, 'write_rows', fail)
    with pytest.raises(ValueError): collector.finish('turn-1', 8, time.monotonic()+2)
    assert not bridge.forward.acknowledged() and collector.failed


def test_timeout_and_cancel_prevent_future_watch(tmp_path):
    collector, bridge, _, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    with pytest.raises(TimeoutError): collector.finish('turn-1', 8, time.monotonic()-.1)
    with pytest.raises(ValueError): collector.begin('turn-2', proof, 8)
    assert bridge.watching == 'turn-1'


def test_manifest_failure_does_not_ack_completion(tmp_path, monkeypatch):
    import drift.serving.mcdma_turn_outbox as module
    collector, bridge, native, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    output(native, 'turn-1')
    def fail(*args): raise OSError('synthetic full disk')
    monkeypatch.setattr(module, 'write_manifest', fail)
    with pytest.raises(ValueError): collector.finish('turn-1', 8, time.monotonic()+2)
    assert bridge.in_flight == 'complete' and not bridge.forward.acknowledged()


def test_empty_completion_still_requires_expected_source_start(tmp_path):
    collector, bridge, native, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    folder = native / 'turn-1'
    folder.mkdir()
    (folder / 'finished').write_text(json.dumps(dict(failed='', writes_scheduled=0, tap_count=0,
                                                   source_start=4, source_stop=4)))
    with pytest.raises(ValueError): collector.finish('turn-1', 8, time.monotonic()+2)
    assert not bridge.forward.acknowledged()


@pytest.mark.parametrize('deadline', [True, float('nan'), float('inf')])
def test_invalid_deadline_never_consumes(tmp_path, deadline):
    collector, bridge, native, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    output(native, 'turn-1')
    bridge.publish_taps()
    with pytest.raises(ValueError): collector.finish('turn-1', 8, deadline)
    assert collector.failed and not bridge.forward.acknowledged()


def test_parent_cancellation_blocks_watch(tmp_path):
    from drift.exchange.lifetime import request_scope
    collector, bridge, _, proof = setup(tmp_path)
    def cancelled(): raise TimeoutError('synthetic cancellation')
    with request_scope(cancelled), pytest.raises(TimeoutError):
        collector.begin('turn-1', proof, 8)
    assert collector.failed and bridge.watching is None
