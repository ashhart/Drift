"""Receiver admission and poison regression tests use synthetic caches only."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_vllm_glm53_inject import connector, LAYERS


def studio_handle(state, append):
    tree = ast.parse(Path('scripts/live/studio_drift_worker.py').read_text())
    namespace = dict(S=state, np=np, append_entries=append, rope=None, INDEX_DIM=8,
                     KV=[0, 1], KV_LAYOUTS={0: (2, 4), 1: (2, 4)}, mx=NS(bfloat16='bf16', eval=lambda *values: None))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, 'module', '') == 'drift.serving.live_receiver_studio':
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<worker-import>', 'exec'), namespace)
        if isinstance(node, ast.FunctionDef) and node.name == 'handle':
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<worker-handle>', 'exec'), namespace)
    return namespace['handle']


def studio_state():
    cache = [NS(offset=3, keys=np.zeros((1, 2, 3, 4)), values=np.zeros((1, 2, 3, 4))) for _ in range(2)]
    return dict(cache=cache, reserve=8, foreign_pos=0, own_pos=3, slot_pos=[0, 1, 2], own_slots=[0, 1, 2], span=[3, 11], span_next=3)


def studio_memory(path, fault=None):
    arrays = {f'{kind}{layer}': np.ones((2, 2, 4), np.float32) for layer in range(2) for kind in 'kv'}
    if fault == 'missing':
        del arrays['k1']; del arrays['v1']
    if fault == 'nonfinite':
        arrays['v1'][0, 0, 0] = np.nan
    if fault == 'shape':
        arrays['v1'] = np.ones((2, 3, 4), np.float32)
    np.savez(path, **arrays)


@pytest.mark.parametrize('fault', ['missing', 'nonfinite', 'shape'])
def test_studio_rejects_every_layer_before_cache_mutation(tmp_path, fault):
    state, calls = studio_state(), []
    handle = studio_handle(state, lambda *args, **kwargs: calls.append(True))
    path = tmp_path / 'memory.npz'
    studio_memory(path, fault)
    with pytest.raises(ValueError):
        handle({'op': 'append', 'memory': str(path)})
    assert calls == []
    assert state['span_next'] == 3 and state['foreign_pos'] == 0
    assert state['poisoned'] is True
    with pytest.raises(RuntimeError, match='poisoned'):
        handle({'op': 'generate', 'tokens': 0})


def test_studio_partial_write_poison_blocks_retry_and_preserves_cursor(tmp_path):
    state = studio_state()
    def partial_write(cache, *args, **kwargs):
        cache[0].offset += 2
        raise RuntimeError('synthetic device failure')
    handle = studio_handle(state, partial_write)
    path = tmp_path / 'memory.npz'
    studio_memory(path)
    with pytest.raises(RuntimeError):
        handle({'op': 'append', 'memory': str(path)})
    assert state['span_next'] == 3 and state['foreign_pos'] == 0
    assert state['poisoned'] is True
    with pytest.raises(RuntimeError, match='poisoned'):
        handle({'op': 'append', 'memory': str(path)})


@pytest.mark.parametrize('fault', ['missing', 'shape', 'nonfinite'])
def test_glm_live_rejects_before_any_layer_write_and_latches_poison(connector, fault):
    c, root = connector
    folder = root / 'tp-live-in' / 'safe-test'
    folder.mkdir(parents=True)
    arrays = {'l3': np.ones((2, 512), np.float32), 'l7': np.ones((2, 512), np.float32)}
    if fault == 'missing':
        del arrays['l7']
    if fault == 'shape':
        arrays['l7'] = np.ones((2, 511), np.float32)
    if fault == 'nonfinite':
        arrays['l7'][0, 0] = np.nan
    np.savez(folder / '000000.npz', **arrays)
    step = NS(name='safe-test', request_id='r', reserve=4, reserve_start=0, blocks=((0,), (1,)), before=4, after=5, apply=(0,), tap=False, failed='')
    c.meta = NS(inject=[], live=[step], requests=[])
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not any(bool(t.any()) for t in c._kv_caches.values())
    assert (root / 'tp-live-out' / 'safe-test' / 'error.rank0').exists()
    arrays['l7'] = np.ones((2, 512), np.float32)
    np.savez(folder / '000000.npz', **arrays)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not any(bool(t.any()) for t in c._kv_caches.values())
    assert c._live_worker['safe-test']['poisoned'] is True


@pytest.mark.parametrize('fault', ['offset', 'cache_shape'])
def test_studio_rejects_bad_destination_before_mutating_any_layer(tmp_path, fault):
    state, calls = studio_state(), []
    if fault == 'offset':
        state['cache'][1].offset = 2
    else:
        state['cache'][1].keys = np.zeros((1, 3, 3, 4))
    handle = studio_handle(state, lambda *args, **kwargs: calls.append(True))
    path = tmp_path / 'memory.npz'
    studio_memory(path)
    with pytest.raises(ValueError):
        handle({'op': 'append', 'memory': str(path)})
    assert calls == [] and state['poisoned']


@pytest.mark.parametrize('framed', [False, True])
def test_studio_valid_append_commits_cursor_only_after_cache_write(tmp_path, framed):
    state, calls = studio_state(), []
    if not framed:
        del state['span']; del state['span_next']
    def append(cache, entries, rope, positions, index_dim, dtype):
        assert state['foreign_pos'] == 0
        calls.append(positions.tolist())
        for item in cache:
            item.offset += 2
    handle = studio_handle(state, append)
    path = tmp_path / 'memory.npz'
    studio_memory(path)
    report = handle({'op': 'append', 'memory': str(path)})
    assert calls == [[3, 4] if framed else [6, 7]]
    assert report == {'appended': 2, 'foreign_total': 2, 'cache_slots': 5}
    assert state['poisoned'] is False
    assert state.get('span_next', 5) == 5


def test_glm_device_failure_poison_blocks_future_writes_and_taps(connector, monkeypatch):
    c, root = connector
    folder = root / 'tp-live-in' / 'device-test'
    folder.mkdir(parents=True)
    np.savez(folder / '000000.npz', l3=np.ones((2, 512), np.float32), l7=np.ones((2, 512), np.float32))
    step = NS(name='device-test', request_id='r', reserve=4, reserve_start=0, blocks=((0,), (1,)), before=12, after=13, apply=(0,), tap=True, failed='')
    c.meta = NS(inject=[], live=[step], requests=[])
    original = torch.Tensor.__setitem__
    def fail_second(tensor, index, value):
        if tensor is c._kv_caches[LAYERS[1]]:
            raise RuntimeError('synthetic device write failure')
        return original(tensor, index, value)
    monkeypatch.setattr(torch.Tensor, '__setitem__', fail_second)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert c._kv_caches[LAYERS[0]].any() and not c._kv_caches[LAYERS[1]].any()
    assert c._live_worker['device-test']['poisoned']
    assert c._live_worker['device-test']['filled'] == 0
    monkeypatch.setattr(torch.Tensor, '__setitem__', original)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not c._kv_caches[LAYERS[1]].any()
    assert not list((root / 'tp-live-out' / 'device-test').glob('*.npz'))


def test_studio_lazy_device_failure_is_poisoned_before_cursor_commit(tmp_path):
    state = studio_state()
    handle = studio_handle(state, lambda *args, **kwargs: None)
    def fail_settle(*tensors):
        raise RuntimeError('synthetic lazy device failure')
    handle.__globals__['mx'].eval = fail_settle
    path = tmp_path / 'memory.npz'
    studio_memory(path)
    with pytest.raises(RuntimeError, match='lazy device'):
        handle({'op': 'append', 'memory': str(path)})
    assert state['foreign_pos'] == 0 and state['span_next'] == 3 and state['poisoned']


def test_studio_fresh_start_discards_poisoned_cache():
    state = studio_state()
    old_cache = state['cache']
    state['poisoned'] = True
    handle = studio_handle(state, lambda *args, **kwargs: None)
    handle.__globals__['lm'] = NS(make_cache=lambda: [])
    assert handle({'op': 'start', 'reserve': 8}) == {'kv_layers': [0, 1]}
    assert state['cache'] is not old_cache and not state.get('poisoned')
    assert state['foreign_pos'] == 0 and state['slot_pos'] == []


def test_glm_live_error_marker_and_log_exclude_exception_payload(connector, monkeypatch, caplog):
    c, root = connector
    def fail_targets():
        raise RuntimeError('synthetic-private-payload')
    monkeypatch.setattr(c, '_live_targets', fail_targets)
    step = NS(name='safe-error', reserve=4, reserve_start=0, failed='')
    c.meta = NS(inject=[], live=[step], requests=[])
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    marker = (root / 'tp-live-out' / 'safe-error' / 'error.rank0').read_text()
    assert 'poisoned' in marker and 'synthetic-private-payload' not in marker
    assert 'synthetic-private-payload' not in caplog.text


def test_glm_log_preserves_safe_page_diagnostics_and_stack(connector, caplog):
    c, root = connector
    folder = root / 'tp-live-in' / 'page-error'
    folder.mkdir(parents=True)
    np.savez(folder / '000000.npz', l3=np.ones((9, 512), np.float32), l7=np.ones((9, 512), np.float32))
    step = NS(name='page-error', reserve=10, reserve_start=0, blocks=((0,), (1,)),
              before=10, apply=(0,), tap=False, failed='')
    c.meta = NS(inject=[], live=[step], requests=[])
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert 'PAGE_RANGE' in caplog.text
    assert '"start": 0' in caplog.text and '"stop": 9' in caplog.text
    assert '"pages": 2' in caplog.text and '"slots": 4' in caplog.text
    assert '_live_index' in caplog.text
    assert str(root) not in caplog.text
    marker = (root / 'tp-live-out' / 'page-error' / 'error.rank0').read_text()
    assert 'PAGE_RANGE' in marker and 'stack' not in marker


@pytest.mark.parametrize('code,details', [
    ('private-payload', {}), ('PAGE_RANGE', {'private-payload': 1}),
    ('BLOCK_GROUP', {'group': 'private-payload', 'groups': 2}),
    ('BLOCK_GROUP', {'group': True, 'groups': 2}),
])
def test_glm_diagnostics_reject_nonstructural_details(code, details):
    from drift.serving.live_receiver_glm import LiveReceiverError
    with pytest.raises(ValueError):
        LiveReceiverError(code, **details)
