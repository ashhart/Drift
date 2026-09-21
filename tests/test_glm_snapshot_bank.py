"""Bound versioned activation snapshots without changing a captured reader turn."""
import hashlib
from pathlib import Path
import numpy as np
import pytest


def make_bank(tmp_path, **extra):
    from drift.serving.glm_snapshot_bank import SnapshotBank
    incoming = tmp_path / 'incoming'; incoming.mkdir(mode=0o700)
    settings = dict(session='fixture', source_worker='qwen', target_worker='glm', recipe_sha256='a' * 64,
                    layouts={'l3': (512,)}, max_rows=4, max_bytes=16384, max_versions=3, max_total_bytes=32768)
    settings.update(extra)
    return SnapshotBank(incoming, **settings), incoming


def update(root, version, rows=2, **extra):
    path = root / f'input-{version}.npz'
    np.savez(path, l3=np.full((rows, 512), version, dtype=np.float32))
    frame = dict(v=1, session='fixture', version=version, path=path.name, rows=rows,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_worker='qwen', target_worker='glm', recipe_sha256='a' * 64)
    frame.update(extra)
    return frame


def test_captured_version_survives_new_publication_and_incoming_replacement(tmp_path):
    bank, root = make_bank(tmp_path)
    first = update(root, 1); bank.publish(first); captured = bank.capture()
    (root / first['path']).write_bytes(b'replaced incoming')
    second = update(root, 2, rows=3); receipt = bank.publish(second)
    assert captured.version == 1 and bank.capture().version == 2
    bank.validate(captured)
    assert captured.sha256 == first['sha256'] and receipt['sha256'] == second['sha256']
    assert Path(captured.path).stat().st_mode & 0o777 == 0o400
    assert receipt['source_worker'] == 'qwen' and 'path' not in receipt


@pytest.mark.parametrize('change', [dict(source_worker='glm'), dict(target_worker='qwen'), dict(recipe_sha256='b'*64), dict(version=2), dict(rows=3), dict(path='../outside.npz'), dict(sha256='f'*64)])
def test_invalid_update_poisoned_before_version_can_advance(tmp_path, change):
    bank, root = make_bank(tmp_path)
    frame = update(root, 1); frame.update(change)
    with pytest.raises(ValueError): bank.publish(frame)
    with pytest.raises(ValueError): bank.capture()
    assert not list(bank.directory.glob('version-*.npz'))


def test_captured_file_tampering_poisoned_before_reuse(tmp_path):
    bank, root = make_bank(tmp_path); bank.publish(update(root, 1)); captured = bank.capture()
    path = Path(captured.path); path.chmod(0o600); path.write_bytes(b'corrupt')
    with pytest.raises(ValueError): bank.validate(captured)
    with pytest.raises(ValueError): bank.publish(update(root, 2))


def test_stored_version_and_byte_budgets_are_not_reset_by_capture(tmp_path):
    bank, root = make_bank(tmp_path, max_versions=1)
    bank.publish(update(root, 1)); bank.capture()
    with pytest.raises(ValueError): bank.publish(update(root, 2))
    other = tmp_path / 'other'; other.mkdir()
    bank, root = make_bank(other, max_total_bytes=4500)
    bank.publish(update(root, 1)); bank.capture()
    with pytest.raises(ValueError): bank.publish(update(root, 2))


@pytest.mark.parametrize('corruption', ['symlink', 'nonfinite', 'layer'])
def test_bad_activation_file_never_becomes_a_version(tmp_path, corruption):
    bank, root = make_bank(tmp_path)
    frame = update(root, 1); path = root / frame['path']
    if corruption == 'symlink':
        target = tmp_path / 'outside.npz'; path.rename(target); path.symlink_to(target)
    else:
        arrays = {'l3' if corruption == 'nonfinite' else 'l4': np.full((2,512), np.nan if corruption == 'nonfinite' else 1, dtype=np.float32)}
        np.savez(path, **arrays); frame['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): bank.publish(frame)
    assert not list(bank.directory.glob('version-*.npz'))


def test_caller_cannot_rewrite_version_metadata_during_snapshot_copy(tmp_path, monkeypatch):
    import drift.serving.glm_snapshot_bank as module
    bank, root = make_bank(tmp_path); frame = update(root, 1)
    original = module.snapshot
    def copying(source, target, maximum):
        frame['version'] = 2
        return original(source, target, maximum)
    monkeypatch.setattr(module, 'snapshot', copying)
    receipt = bank.publish(frame)
    assert receipt['version'] == 1 and bank.capture().version == 1
