import importlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/omp'))


def test_room_profile_rejects_unbounded_or_linked_owner(tmp_path):
    module = importlib.import_module('native_room_profile')
    identity = dict(session='fresh', worker='glm', model_id='glm', model_sha256='a'*64, translator_sha256='b'*64)
    entry = dict(identity=identity, expected=dict(backend='live', nativeStates=['reconstructed']), experimental_multi_turn=True, memory_mode='no-link', limits=module.LIMITS, context_window=32768)
    module.validate_entry(entry)
    for patch in ({'memory_mode': 'linked'}, {'experimental_multi_turn': False}, {'limits': {**module.LIMITS, 'deadline_ms': 120000}}, {'identity': {**identity, 'model_sha256': ''}}):
        with pytest.raises(ValueError): module.validate_entry({**entry, **patch})


def test_cleanup_requires_exact_successful_release():
    module = importlib.import_module('native_room_evidence')
    receipt = dict(event='worker_terminated', reason='input_eof', child_pid=123, returncode=0, child_reaped=True, process_group_alive=False, term_sent=False, kill_sent=False, wall_seconds=20, input_bytes=12, output_bytes=34)
    assert module.cleanup_ok(receipt)
    for patch in ({'child_reaped': False}, {'process_group_alive': True}, {'returncode': 2}, {'kill_sent': True}, {'wall_seconds': 61}):
        assert not module.cleanup_ok({**receipt, **patch})
    assert not module.cleanup_ok({})


def make_profile(tmp_path):
    from native_room_profile import digest, tree_digest, LIMITS
    artifact = tmp_path/'pinned'; artifact.write_text('fixture pin, no model')
    pin = dict(path=str(artifact), sha256=digest(artifact))
    command = dict(executable='/usr/bin/true', sha256=digest('/usr/bin/true'), args=[], cwd=str(tmp_path), env={'PATH': '/usr/bin:/bin'}, artifacts=[pin])
    workers = [dict(identity=dict(session='fresh-'+name, worker=name, model_id=name, model_sha256='a'*64, translator_sha256='b'*64), expected=dict(backend='live', nativeStates=[state]), experimental_multi_turn=True, memory_mode='no-link', limits=LIMITS, context_window=32768, command=command, artifacts=[pin]) for name, state in [('glm', 'reconstructed'), ('qwen', 'retained')]]
    owner = tmp_path/'owner.json'; owner.write_text(json.dumps(dict(version=1, experimental=True, workers=workers)))
    duo = tmp_path/'duo'; (duo/'src').mkdir(parents=True); (duo/'agents').mkdir(); (duo/'src/omp.ts').write_text('export default ()=>{}'); (duo/'agents/duo-peer.md').write_text('fixture')
    value = dict(version=1, purpose='restricted-no-link-room', owner=dict(path=str(owner), sha256=digest(owner)), omp=dict(path='/usr/bin/true', sha256=digest('/usr/bin/true')), duo=dict(path=str(duo), sha256=tree_digest(duo)), parent='drift-experimental/glm', child='drift-experimental/qwen', cleanup=[dict(worker=name, session='fresh-'+name, command={**command, 'args': [name]}) for name in ('glm', 'qwen')])
    path = tmp_path/'profile.json'; path.write_text(json.dumps(value))
    return path, value


def test_prepare_never_dispatches_and_requires_frozen_prepared_hash(tmp_path, monkeypatch):
    from native_room import run
    from native_room_profile import digest
    import native_room
    profile, _ = make_profile(tmp_path)
    def forbidden(*args, **kwargs): raise AssertionError('must not dispatch')
    monkeypatch.setattr(native_room, 'execute', forbidden)
    monkeypatch.setattr(native_room, 'read_cleanup', forbidden)
    report = run(profile, digest(profile), tmp_path/'evidence')
    assert report['verdict'] == 'PREPARED'
    with pytest.raises(ValueError, match='ROOM_PREPARED_PIN'): run(profile, digest(profile), tmp_path/'evidence', dispatch=True)


def test_room_pins_compaction_off_for_retained_history(tmp_path):
    from native_room_stage import prepare, verify_stage
    from native_room_profile import digest
    _, profile = make_profile(tmp_path)
    prepared = prepare(profile, tmp_path/'prepared')
    config = Path(prepared['env']['PI_CODING_AGENT_DIR'])/'config.yml'
    assert config.read_text() == 'compaction:\n  enabled: false\n'
    assert prepared['pins']['agent/config.yml'] == digest(config)
    verify_stage(prepared)
    config.write_text('compaction:\n  enabled: true\n')
    with pytest.raises(ValueError, match='ROOM_STAGE_PIN'): verify_stage(prepared)


def test_profile_refuses_missing_route_pins_and_oversized_config(tmp_path):
    from native_room_profile import load, digest
    path, value = make_profile(tmp_path)
    assert load(path, digest(path))[0]['parent'] == value['parent']
    value['child'] = value['parent']; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='ROOM_ROUTE'): load(path, digest(path))
    path.write_text(' '*65537)
    with pytest.raises(ValueError, match='ROOM_SIZE'): load(path, digest(path))


def test_controller_discards_output_and_kills_descendants(tmp_path):
    from native_room_process import execute
    result = execute(['/usr/bin/python3', '-c', 'print("synthetic output")'], cwd=tmp_path, env={'PATH':'/usr/bin:/bin'})
    assert result['exit_code'] == 0 and result['output_joined']
    assert 'synthetic output' not in json.dumps(result)
    result = execute(['/usr/bin/python3', '-c', 'import subprocess;subprocess.Popen(["/bin/sleep","30"])'], cwd=tmp_path, env={'PATH':'/usr/bin:/bin'})
    assert result['descendants_killed'] and result['output_joined']


def test_cleanup_preflight_rejects_old_receipt_and_never_returns_text(tmp_path, monkeypatch):
    import native_room_evidence as evidence
    from types import SimpleNamespace
    item = dict(worker='glm', session='fresh', command=dict(executable='/usr/bin/true', args=[], cwd=str(tmp_path), env={}))
    monkeypatch.setattr(evidence.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b'{}', stderr=b''))
    with pytest.raises(ValueError, match='STALE'): evidence.read_cleanup(item, fresh=True)
    monkeypatch.setattr(evidence.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout=b'', stderr=b''))
    assert evidence.read_cleanup(item, fresh=True) is None
    monkeypatch.setattr(evidence.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b'{"private_text":"never expose"}', stderr=b''))
    result = evidence.read_cleanup(item, fresh=False)
    assert not result['passed'] and 'never expose' not in json.dumps(result)


def test_missing_cleanup_fails_even_after_successful_controller(tmp_path, monkeypatch):
    import native_room as room
    from native_room_profile import digest
    profile, _ = make_profile(tmp_path)
    ready = room.run(profile, digest(profile), tmp_path/'evidence')
    monkeypatch.setattr(room, 'execute', lambda *args, **kwargs: dict(exit_code=0, output_joined=True))
    def read(item, fresh):
        if fresh: return None
        raise ValueError('missing')
    monkeypatch.setattr(room, 'read_cleanup', read)
    result = room.run(profile, digest(profile), tmp_path/'evidence', dispatch=True, prepared_sha256=ready['prepared_sha256'])
    assert result['verdict'] == 'FAILED' and result['errors'] == ['CLEANUP_UNVERIFIED']*2
    assert (tmp_path/'evidence/report.json').exists()
    with pytest.raises(FileExistsError): room.run(profile, digest(profile), tmp_path/'evidence', dispatch=True, prepared_sha256=ready['prepared_sha256'])
