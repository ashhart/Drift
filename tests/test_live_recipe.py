"""Live recipe loading uses tiny synthetic artifacts, never experiment weights."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]


def fixture_manifest(tmp_path):
    base = tmp_path / 'base.npz'
    np.savez(base, g_mean=np.zeros(2, np.float32), W3=np.ones((2, 4), np.float32),
             b3=np.zeros(4, np.float32), gain3=np.ones(4, np.float32))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    fan = tmp_path / 'fan.npz'
    np.savez(fan, meta=json.dumps({'base_sha256': sha(base), 'margin': 0}), basis=np.ones((2, 1), np.float32),
             count_w=np.zeros((1, 2), np.float32), count_b=np.array([0, 2], np.float32), R1=np.ones((1, 4), np.float32))
    correction = tmp_path / 'correction.safetensors'
    save_file({'A': np.ones((2, 1), np.float32), 'B': np.ones((1, 4), np.float32),
               'loud': np.zeros((1, 2), np.float32)}, str(correction))
    paths = dict(forward_base=base, forward_fanout=fan, correction=correction)
    data = {'artifacts': {key: str(path) for key, path in paths.items()},
            'sha256': {key: sha(path) for key, path in paths.items()}, 'forward_gain_power': 1.5}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(data))
    return manifest, data


def live_lib(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tokenizers', None)
    spec = importlib.util.spec_from_file_location('synthetic_livelib', ROOT / 'scripts/live/livelib.py')
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    monkeypatch.setattr(lib, 'GLM_LAYERS', (3,))
    monkeypatch.setattr(lib, 'QWEN_LAYERS', (3,))
    return lib


def test_livelib_accepts_an_explicit_pinned_v4_recipe(tmp_path, monkeypatch):
    lib = live_lib(monkeypatch)
    manifest, data = fixture_manifest(tmp_path)
    reader = lib.load_reader(Path(data['artifacts']['forward_base']), recipe='v4', manifest=manifest,
                             kv_heads=1, head_dim=2)
    assert reader.kv_heads == 1 and reader.head_dim == 2
    result = reader.read({3: np.ones((1, 2), np.float32)}, 1.5)
    assert result[3][0].shape == (2, 1, 2)
    np.testing.assert_array_equal(result[3][0][:, 0], [[6, 6], [4, 4]])
    assert reader.metadata['live_qualification'] == 'BLOCKED'
    assert reader.metadata['kind'] == 'v4'
    assert reader.metadata['sha256'] == data['sha256']


@pytest.mark.parametrize('kind,rows', [('base', 1), ('fanout', 2), ('v4', 2)])
def test_all_explicit_recipes_use_frozen_gains_and_selected_hashes(tmp_path, monkeypatch, kind, rows):
    lib = live_lib(monkeypatch)
    manifest, data = fixture_manifest(tmp_path)
    reader = lib.load_reader(None, recipe=kind, manifest=manifest, kv_heads=1, head_dim=2)
    assert reader.read({3: np.ones((1, 2), np.float32)}, 1.5)[3][0].shape == (rows, 1, 2)
    assert reader.metadata['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    keys = {'forward_base'} | ({'forward_fanout'} if kind != 'base' else set()) | ({'correction'} if kind == 'v4' else set())
    assert set(reader.metadata['sha256']) == keys
    with pytest.raises(ValueError, match='gain'):
        reader.read({3: np.ones((1, 2), np.float32)}, 0.5)


@pytest.mark.parametrize('key', ['forward_base', 'forward_fanout', 'correction'])
def test_artifact_tampering_is_rejected_before_reader_use(tmp_path, monkeypatch, key):
    lib = live_lib(monkeypatch)
    manifest, data = fixture_manifest(tmp_path)
    with Path(data['artifacts'][key]).open('ab') as artifact:
        artifact.write(b'tampered')
    with pytest.raises(ValueError, match='SHA-256'):
        lib.load_reader(None, recipe='v4', manifest=manifest, kv_heads=1, head_dim=2)


def test_legacy_load_stays_one_to_one_without_manifest(tmp_path, monkeypatch):
    lib = live_lib(monkeypatch)
    _, data = fixture_manifest(tmp_path)
    reader = lib.load_reader(Path(data['artifacts']['forward_base']), kv_heads=1, head_dim=2)
    assert reader.read({3: np.ones((1, 2), np.float32)}, 1.5)[3][0].shape == (1, 1, 2)
    with pytest.raises(ValueError, match='explicit'):
        lib.load_reader(Path(data['artifacts']['forward_base']), manifest=tmp_path / 'unused')


@pytest.mark.parametrize('field,value', [('basis', np.full((2, 1), np.nan)),
                                        ('count_w', np.zeros((2, 2))),
                                        ('R1', np.zeros((1, 5)))])
def test_invalid_fanout_parameters_fail_even_with_updated_file_pin(tmp_path, monkeypatch, field, value):
    lib = live_lib(monkeypatch)
    manifest, data = fixture_manifest(tmp_path)
    path = Path(data['artifacts']['forward_fanout'])
    with np.load(path) as old:
        arrays = {key: old[key] for key in old.files}
    arrays[field] = value
    np.savez(path, **arrays)
    data['sha256']['forward_fanout'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='fan-out'):
        lib.load_reader(None, recipe='fanout', manifest=manifest, kv_heads=1, head_dim=2)
