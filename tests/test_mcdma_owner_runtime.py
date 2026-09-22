"""Compose the owner entry point with pinned connections and real collectors."""
import time

import pytest

from tests.test_mcdma_connections import fixture
from test_glm_owner_worker import configuration
from test_glm_owner_worker import root


def config(root):
    root.chmod(0o700)
    value = configuration(root)
    value['restoration_transport'] = 'mcdma'
    output = root / 'exports'
    output.mkdir(mode=0o700)
    value['outbox'] = dict(memory_root=str(output), source_worker='glm', target_worker='qwen',
                           recipe_sha256=value['owner_bank']['recipe_sha256'], max_raw_rows=32,
                           max_raw_bytes=65536, max_rows=8, max_bytes=32768, max_publications=4)
    return value


def test_owner_gets_real_factories_and_all_connections_close(tmp_path):
    from drift.serving.mcdma_owner_runtime import run_owner
    from drift.serving.mcdma_turn_outbox import McdmaTurnOutbox
    from drift.serving.mcdma_restoration import RestorationPublication
    value = config(tmp_path)
    targets, _, opened, opener = fixture()
    def serve(configuration, **kwargs):
        assert configuration is value
        assert kwargs['memory_root'] == tmp_path
        collector = kwargs['outbox_factory'](value['outbox'], {'l3': (512,)}, max_turns=2)
        assert isinstance(collector, McdmaTurnOutbox)
        assert collector.reader.expected == 1
        assert len(opened) == 3
        recipe = configuration['restoration']
        publication = kwargs['publication_factory'](recipe['snapshot_path'], recipe['snapshot_sha256'],
                                                     'request-one', lambda: 1)
        assert isinstance(publication, RestorationPublication)
        assert publication.publisher is collector.publisher
        collector.close()
        return 0
    assert run_owner(value, targets, opener, memory_root=tmp_path, serve=serve) == 0
    assert len(opened) == 3 and all(conn.closed for conn in opened)


@pytest.mark.parametrize('outcome', ['raise', 'failed', 'cancel'])
def test_owner_failure_still_closes_native_handles(tmp_path, outcome):
    from drift.serving.mcdma_owner_runtime import run_owner
    value = config(tmp_path)
    targets, _, opened, opener = fixture()
    def serve(configuration, **kwargs):
        kwargs['outbox_factory'](value['outbox'], {'l3': (512,)}, max_turns=2)
        if outcome == 'raise': raise RuntimeError('synthetic failure')
        if outcome == 'cancel': raise KeyboardInterrupt
        return 2
    if outcome == 'failed':
        assert run_owner(value, targets, opener, memory_root=tmp_path, serve=serve) == 2
    else:
        with pytest.raises(KeyboardInterrupt if outcome == 'cancel' else RuntimeError):
            run_owner(value, targets, opener, memory_root=tmp_path, serve=serve)
    assert len(opened) == 3 and all(conn.closed for conn in opened)


@pytest.mark.parametrize('fault', ['ssh', 'no_link', 'no_outbox', 'wrong_target', 'deadline', 'layers', 'remote_http'])
def test_invalid_runtime_never_opens_transport(tmp_path, fault):
    from drift.serving.mcdma_owner_runtime import run_owner
    value = config(tmp_path)
    targets, _, opened, opener = fixture()
    if fault == 'ssh': value['restoration_transport'] = 'ssh'
    elif fault == 'no_link': value['memory_mode'] = 'no_link'
    elif fault == 'no_outbox': del value['outbox']
    elif fault == 'wrong_target': value['outbox']['target_worker'] = 'third-worker'
    elif fault == 'deadline': value['limits']['deadline_ms'] = 600001
    elif fault == 'remote_http': value['base_url'] = 'http://192.0.2.1:8888'
    else: value['restoration']['layers'] = [3, 3]
    with pytest.raises(ValueError):
        run_owner(value, targets, opener, memory_root=tmp_path, serve=lambda *a, **kw: pytest.fail('dispatched'))
    assert opened == []


def test_no_work_does_not_claim_or_stamp_the_targets(tmp_path):
    from drift.serving.mcdma_owner_runtime import run_owner
    targets, regions, opened, opener = fixture()
    assert run_owner(config(tmp_path), targets, opener, memory_root=tmp_path, serve=lambda *a, **kw: 2) == 2
    assert opened == [] and not any(region.puts for region in regions)


def test_owner_startup_cannot_reset_transport_lifetime(tmp_path):
    from drift.serving.mcdma_owner_runtime import run_owner
    targets, _, opened, opener = fixture()
    value = config(tmp_path)
    ticks = [time.monotonic()]
    def serve(configuration, **kwargs):
        ticks[0] += 5
        kwargs['outbox_factory'](value['outbox'], {'l3': (512,)}, max_turns=2)
    with pytest.raises(TimeoutError):
        run_owner(value, targets, opener, memory_root=tmp_path, serve=serve, clock=lambda: ticks[0])
    assert opened == []


def test_real_owner_protocol_constructs_bank_collector_and_closes(root, monkeypatch):
    import io
    import json
    import os
    from types import SimpleNamespace
    from drift.serving.glm_session import GlmSession
    from drift.serving.mcdma_owner_runtime import run_owner
    import drift.serving.glm_restore_factory as factory
    value = config(root)
    targets, _, opened, opener = fixture()
    transport = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(factory, 'unlinked_factory', lambda _: GlmSession(transport, model='fixture'))
    frames = [{'v': 1, 'session': 'owned', 'worker': 'glm', 'seq': 1, 'op': 'open',
               'payload': {**value['pins'], 'limits': value['limits'], 'system_prompt': [], 'tools': []}},
              {'v': 1, 'session': 'owned', 'worker': 'glm', 'seq': 2, 'op': 'close', 'payload': {}}]
    incoming, writer = os.pipe()
    sink = io.StringIO()
    try:
        os.write(writer, b''.join(json.dumps(frame).encode() + b'\n' for frame in frames))
        os.close(writer)
        assert run_owner(value, targets, opener, memory_root=root, input_fd=incoming, sink=sink) == 0
    finally:
        os.close(incoming)
    assert [json.loads(line)['op'] for line in sink.getvalue().splitlines()] == ['opened', 'closed']
    result = json.loads((root / 'bank.json').read_bytes())
    assert result['status'] == 'PASSED' and result['owner_thread_joined']
    assert len(opened) == 3 and all(conn.closed for conn in opened)
    assert not (root / 'owner.sock').exists()
