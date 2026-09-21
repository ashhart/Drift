"""Reject unbounded, replayed or unverified publication transport before mutations."""
import hashlib
from pathlib import Path
import subprocess

import pytest

from drift.serving.live_publication_transport import PublicationTransport


def transport(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'synthetic')
    folder = tmp_path / 'tp-live-in/one'
    folder.mkdir(parents=True)
    item = PublicationTransport(source, hashlib.sha256(source.read_bytes()).hexdigest(), 'one', 'peer', lambda: None, root=tmp_path)
    return item, folder


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_expired_stage_or_publish_never_starts_io(tmp_path, monkeypatch, timeout):
    item, folder = transport(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('unexpected SSH'))
    with pytest.raises(TimeoutError): item.stage(timeout=timeout)
    item.staged = True
    with pytest.raises(TimeoutError): item.publish(timeout=timeout)
    assert list(folder.iterdir()) == []


def test_peer_failure_does_not_create_head_publication(tmp_path, monkeypatch):
    item, folder = transport(tmp_path)
    def failed(*args, **kwargs): raise subprocess.CalledProcessError(2, ['synthetic'])
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(subprocess.CalledProcessError): item.stage(timeout=1)
    assert list(folder.iterdir()) == []


def test_head_publication_cannot_replace_existing_evidence(tmp_path, monkeypatch):
    item, folder = transport(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: None)
    item.stage(timeout=1)
    final = folder / '000000.npz'
    final.write_bytes(b'preserve')
    with pytest.raises(FileExistsError): item.publish(timeout=1)
    assert final.read_bytes() == b'preserve'
