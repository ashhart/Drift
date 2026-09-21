"""Synthetic faults exercise the live connector's real cache commit callsite."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_vllm_glm53_inject import LAYERS, connector


def publication(connector, rows=2):
    c, root = connector
    folder = root / 'tp-live-in' / 'commit-test'
    folder.mkdir(parents=True)
    arrays = {f'l{layer}': np.arange(rows * 512, dtype=np.float32).reshape(rows, 512) + 1
              for layer in (3, 7)}
    np.savez(folder / '000000.npz', **arrays)
    step = NS(name='commit-test', reserve=rows, reserve_start=0,
              blocks=((0,), (1,)), before=rows, after=rows + 1,
              apply=(0,), tap=False, failed='')
    c.meta = NS(inject=[], live=[step], requests=[])
    return c, root


@pytest.mark.parametrize('fault', ['dropped', 'corrupt'])
def test_no_receipt_when_device_write_does_not_match_publication(connector, monkeypatch, caplog, fault):
    c, root = publication(connector)
    original = torch.Tensor.__setitem__

    def defective_write(tensor, index, values):
        if tensor is c._kv_caches[LAYERS[-1]]:
            if fault == 'dropped':
                return
            values = values.clone()
            values[0, 0] ^= 1
        return original(tensor, index, values)

    monkeypatch.setattr(torch.Tensor, '__setitem__', defective_write)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not list((root / 'tp-live-out' / 'commit-test').glob('ack.*'))
    assert c._live_worker['commit-test']['poisoned']
    assert c._live_worker['commit-test']['filled'] == 0
    assert 'next_sequence' not in c._live_worker['commit-test']
    marker = (root / 'tp-live-out' / 'commit-test' / 'error.rank0').read_text()
    assert marker == 'RuntimeError: live receiver failed; session poisoned'
    assert 'tensor(' not in caplog.text and 'array(' not in caplog.text
    monkeypatch.setattr(torch.Tensor, '__setitem__', original)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not list((root / 'tp-live-out' / 'commit-test').glob('ack.*'))


def test_aliased_destination_slots_rejected_before_any_layer_write(connector, monkeypatch):
    c, root = publication(connector)
    original = c._live_index

    def aliased(part, blocks, start, stop):
        page, slot = original(part, blocks, start, stop)
        if part is c._kv_caches[LAYERS[-1]]:
            page[:] = page[0]
            slot[:] = slot[0]
        return page, slot

    monkeypatch.setattr(c, '_live_index', aliased)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not any(bool(cache.any()) for cache in c._kv_caches.values())
    assert c._live_worker['commit-test']['poisoned']
    assert not list((root / 'tp-live-out' / 'commit-test').glob('ack.*'))


@pytest.mark.parametrize('fault', ['negative_slot', 'past_slot', 'short', 'float'])
def test_bad_destination_indices_reject_every_layer_before_write(connector, monkeypatch, fault):
    c, root = publication(connector)
    original = c._live_index

    def bad_index(part, blocks, start, stop):
        page, slot = original(part, blocks, start, stop)
        if part is c._kv_caches[LAYERS[-1]]:
            if fault == 'negative_slot':
                slot[0] = -1
            elif fault == 'past_slot':
                slot[0] = part.shape[1]
            elif fault == 'short':
                slot = slot[:-1]
            else:
                page = page.float()
        return page, slot

    monkeypatch.setattr(c, '_live_index', bad_index)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert not any(bool(cache.any()) for cache in c._kv_caches.values())
    assert c._live_worker['commit-test']['poisoned']
    assert not list((root / 'tp-live-out' / 'commit-test').glob('ack.*'))


def test_verified_commit_preserves_rope_and_every_native_slot(connector):
    from drift.serving.glm53_handoff import pack_fp8_ds_mla
    c, root = publication(connector)
    for part in c._kv_caches.values():
        part.fill_(37)
    before = {name: part.clone() for name, part in c._kv_caches.items()}
    c.wait_for_save()
    assert len(list((root / 'tp-live-out' / 'commit-test').glob('ack.*'))) == 1
    assert c._live_worker['commit-test']['filled'] == 2
    expected = torch.from_numpy(pack_fp8_ds_mla(np.arange(1024, dtype=np.float32).reshape(2, 512) + 1))
    for name, part in c._kv_caches.items():
        wanted = before[name]
        if name in LAYERS:
            wanted[2, :2, :528] = expected
        assert torch.equal(part, wanted)
