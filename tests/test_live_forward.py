"""Synthetic fan-out publications exercise the coordinator's real tap function."""
import ast
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from drift.translate.fanout import FanoutReader
from drift.translate.stacked import StackedReader

ROOT = Path(__file__).resolve().parents[1]


def fan_reader():
    base = StackedReader((3,), (3,), 1, 2, np.zeros(2, np.float32),
                         {3: np.ones((2, 4), np.float32)}, {3: np.zeros(4, np.float32)},
                         {3: np.ones(4, np.float32)}, {}, 'synthetic')
    return FanoutReader(base, np.ones((2, 1), np.float32), np.zeros((1, 2), np.float32),
                        np.array([0, 2], np.float32), 0, {1: np.ones((1, 4), np.float32)}, {}, 'synthetic')


def tap_context(tmp_path, *, reserve=8, maximum=4):
    folder = tmp_path / 'glm_taps'
    folder.mkdir()
    for seq in range(2):
        np.savez(folder / f'{seq:06d}.npz', start=seq, stop=seq + 1, l3=np.ones((1, 2), np.float32))
    deliveries, logs = [], []
    def publish(path, rows, upload):
        upload()
        deliveries.append((path, rows))
    namespace = dict(np=np, state={'glm_done': True, 'glm_first': 0, 'forward_reserved_used': 0},
                     args=NS(out=tmp_path, max_publication_rows=maximum, gain_forward=1.5, reserve=reserve),
                     stopping=threading.Event(), GLM_LAYERS=(3,), SPARK='fake', STUDIO='fake', remote='fake', run='fake',
                     glm_layout={'l3': (2,)}, qwen_layout={'k3': (1, 2), 'v3': (1, 2)}, fwd=fan_reader(),
                     pull=lambda *args: None, push=lambda *args: None, check_failure=lambda: None,
                     pending_memory=NS(publish=publish), rank_delivery=NS(health=lambda: None),
                     log=lambda *args, **kw: logs.append(kw))
    tree = ast.parse((ROOT / 'scripts/live/drift_loop.py').read_text())
    selected = [node for node in tree.body if
                isinstance(node, ast.ImportFrom) and (node.module or '').startswith('drift.serving.') or
                isinstance(node, ast.FunctionDef) and node.name == 'glm_tap_thread']
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<live-tap>', 'exec'), namespace)
    return namespace, deliveries, logs


def test_fanout_source_cursor_and_reader_rows_are_distinct(tmp_path):
    context, deliveries, logs = tap_context(tmp_path)
    context['glm_tap_thread']()
    assert [rows for _, rows in deliveries] == [2, 2]
    assert [record['positions'] for record in logs] == [[0, 1], [1, 2]]
    assert [record['reader_rows'] for record in logs] == [2, 2]
    assert context['state']['forward_reserved_used'] == 4
    for seq in range(2):
        with np.load(tmp_path / f'to_qwen_{seq:06d}.npz') as memory:
            assert memory['k3'].shape == memory['v3'].shape == (2, 1, 2)


def test_rank_failure_prevents_forward_fanout_delivery(tmp_path):
    from drift.serving.live_rank_delivery import RankDeliveryError
    context, deliveries, _ = tap_context(tmp_path)
    def fail_rank():
        raise RankDeliveryError('rank reported failure')
    context['rank_delivery'].health = fail_rank
    with pytest.raises(RankDeliveryError):
        context['glm_tap_thread']()
    assert deliveries == [] and context['state']['forward_reserved_used'] == 0


@pytest.mark.parametrize('reserve,maximum', [(1, 4), (8, 1)])
def test_fanout_expansion_is_bounded_before_delivery(tmp_path, reserve, maximum):
    context, deliveries, _ = tap_context(tmp_path, reserve=reserve, maximum=maximum)
    with pytest.raises(ValueError):
        context['glm_tap_thread']()
    assert deliveries == []
    assert not list(tmp_path.glob('to_qwen_*.npz'))



def test_forward_reserve_counts_cumulative_emitted_rows(tmp_path):
    context, deliveries, _ = tap_context(tmp_path, reserve=3)
    with pytest.raises(ValueError, match='reserve'):
        context['glm_tap_thread']()
    assert [rows for _, rows in deliveries] == [2]
    assert context['state']['forward_reserved_used'] == 2
    assert not (tmp_path / 'to_qwen_000001.npz').exists()


@pytest.mark.parametrize('fault', ['unequal', 'missing', 'nonfinite', 'overflow'])
def test_every_emitted_layer_is_checked_before_delivery(tmp_path, monkeypatch, fault):
    context, deliveries, _ = tap_context(tmp_path)
    original = FanoutReader.read
    def broken(reader, latents, gain):
        entries = original(reader, latents, gain)
        k, v = entries[3]
        if fault == 'unequal':
            entries[3] = (k, v[:1])
        elif fault == 'missing':
            entries.clear()
        else:
            v[0, 0, 0] = np.nan if fault == 'nonfinite' else 1e10
        return entries
    monkeypatch.setattr(FanoutReader, 'read', broken)
    with pytest.raises(ValueError):
        context['glm_tap_thread']()
    assert deliveries == []
    assert not list(tmp_path.glob('to_qwen_*.npz'))


def test_expansion_bound_is_checked_before_translation_allocates(tmp_path, monkeypatch):
    context, deliveries, _ = tap_context(tmp_path, maximum=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('oversized translation was allocated')
    monkeypatch.setattr(FanoutReader, 'read', forbidden)
    with pytest.raises(ValueError, match='limit'):
        context['glm_tap_thread']()
    assert deliveries == []



def test_emitted_wire_byte_limit_precedes_translation(tmp_path, monkeypatch):
    from drift.serving import live_forward
    context, deliveries, _ = tap_context(tmp_path)
    monkeypatch.setattr(live_forward, 'MAX_BYTES', 1)
    with pytest.raises(ValueError, match='byte limit'):
        context['glm_tap_thread']()
    assert deliveries == []
    assert not list(tmp_path.glob('to_qwen_*.npz'))
