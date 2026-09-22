import json
import time

import pytest

from drift.transcript.export import extract
from test_transcript_pipeline import export_blob


def test_arena_export_matches_file_export(tmp_path):
    blob = export_blob([1, 2], 'session')
    arena = tmp_path / 'arena'
    arena.write_bytes(b'\0' * 4096 + blob)
    ready = tmp_path / 'rank0.ready'
    ready.write_text(json.dumps({'mode': 'arena', 'offset': 4096, 'length': len(blob),
                                 'arena_inode': arena.stat().st_ino}))
    result = extract(tmp_path, 'session', [1, 2], time.monotonic() + 1, arena=arena)
    assert all(value.shape == (2, 512) for value in result.values())
    ready.write_text(json.dumps({'mode': 'arena', 'offset': 4096, 'length': len(blob), 'arena_inode': 0}))
    with pytest.raises(ValueError, match='BOUNDS'):
        extract(tmp_path, 'session', [1, 2], time.monotonic() + 1, arena=arena)


def test_missing_export_has_finite_wait(tmp_path):
    with pytest.raises(TimeoutError):
        extract(tmp_path, 'session', [1], time.monotonic() + .01)


def test_symlink_export_refused(tmp_path):
    blob = export_blob([1], 'session')
    source = tmp_path / 'source'
    source.write_bytes(blob)
    (tmp_path / 'rank0.bin').symlink_to(source)
    (tmp_path / 'rank0.ready').write_text(json.dumps({'mode': 'file', 'bytes': len(blob)}))
    with pytest.raises(ValueError, match='PATH'):
        extract(tmp_path, 'session', [1], time.monotonic() + 1)
