"""Verify both native artifacts before executing either one."""
import hashlib

import pytest


def pin(path, data):
    path.write_bytes(data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def artifacts(tmp_path):
    binding = b'def open(peer, src=None):\n return (_load(), peer, src)\n'
    return {'binding': pin(tmp_path / 'binding.py', binding),
            'library': pin(tmp_path / 'library.so', b'synthetic-library')}


def test_pinned_binding_only_loads_the_pinned_library(tmp_path, monkeypatch):
    from drift.serving.mcdma_native_loader import load_opener
    import ctypes
    spec = artifacts(tmp_path)
    calls = []
    def load(path):
        calls.append(path)
        return 'synthetic-library'
    monkeypatch.setattr(ctypes, 'CDLL', load)
    opener = load_opener(**spec)
    assert opener('192.0.2.1', src='192.0.2.10') == ('synthetic-library', '192.0.2.1', '192.0.2.10')
    assert calls == [spec['library']['path']]


@pytest.mark.parametrize('artifact', ['binding', 'library'])
@pytest.mark.parametrize('fault', ['hash', 'symlink', 'writable', 'missing'])
def test_bad_pin_never_loads_native_code(tmp_path, monkeypatch, artifact, fault):
    from drift.serving.mcdma_native_loader import load_opener
    from pathlib import Path
    import ctypes
    spec = artifacts(tmp_path)
    path = Path(spec[artifact]['path'])
    if fault == 'hash': spec[artifact]['sha256'] = '0' * 64
    elif fault == 'symlink':
        link = tmp_path / 'link'
        link.symlink_to(path)
        spec[artifact]['path'] = str(link)
    elif fault == 'writable': path.chmod(0o666)
    else: path.unlink()
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a: pytest.fail('native library loaded before validation'))
    with pytest.raises((ValueError, OSError)):
        load_opener(**spec)


def test_invalid_library_is_checked_before_binding_execution(tmp_path, monkeypatch):
    from drift.serving.mcdma_native_loader import load_opener
    import ctypes
    spec = artifacts(tmp_path)
    spec['binding'] = pin(tmp_path / 'raises.py', b'raise RuntimeError("binding was executed")\n')
    spec['library']['sha256'] = '0' * 64
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a: pytest.fail('native library loaded'))
    with pytest.raises(ValueError): load_opener(**spec)
