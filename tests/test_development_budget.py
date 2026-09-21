import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts/omp'))


def test_longer_api_budget_is_explicit_and_room_stays_frozen(tmp_path):
    from test_native_room import make_profile
    from native_room_profile import load, digest
    path, profile = make_profile(tmp_path)
    profile['purpose'] = 'restricted-api-development'
    owner_path = Path(profile['owner']['path']); owner = json.loads(owner_path.read_text())
    for worker in owner['workers']:
        worker['limits'].update(max_turns=10, deadline_ms=170000)
    owner_path.write_text(json.dumps(owner)); profile['owner']['sha256'] = digest(owner_path)
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='ROOM_PROFILE'): load(path, digest(path))
    assert load(path, digest(path), development=True)[0]['purpose'] == profile['purpose']
    owner['workers'][0]['limits']['deadline_ms'] += 1
    owner_path.write_text(json.dumps(owner)); profile['owner']['sha256'] = digest(owner_path); path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='ROOM_LIMIT'): load(path, digest(path), development=True)


def test_api_stage_pins_extended_deadline_without_changing_default(tmp_path):
    from test_native_room import make_profile
    from development_stage import prepare
    _, profile = make_profile(tmp_path)
    profile['purpose'] = 'restricted-api-development'
    prepared = prepare(profile, tmp_path/'stage', Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'), 49276)
    command = prepared['command']
    assert command[command.index('--max-time')+1] == '170'


def test_extended_receipts_require_explicit_limits_and_cannot_overrun():
    from native_room_evidence import cleanup_ok, room_ok
    receipt = dict(event='worker_terminated', reason='input_eof', returncode=0, child_reaped=True,
                   process_group_alive=False, term_sent=False, kill_sent=False, wall_seconds=150)
    assert not cleanup_ok(receipt)
    assert cleanup_ok(receipt, max_seconds=180)
    assert not cleanup_ok({**receipt, 'wall_seconds': 181}, max_seconds=180)
    facts = dict(errors=[], blocked=0, task_calls=1, hub_calls=1, todo_calls=1,
                 workers={'glm': dict(turns=8, output_tokens=4000, input_tokens=6000, shutdown=True)})
    assert not room_ok(facts, ['glm'])
    assert room_ok(facts, ['glm'], max_turns=10)
    assert not room_ok({**facts, 'workers': {'glm': {**facts['workers']['glm'], 'turns': 11}}}, ['glm'], max_turns=10)


def test_reconstructed_development_allowance_can_cover_ten_declared_windows(tmp_path):
    from test_native_room import make_profile
    from native_room_profile import validate_entry
    _,profile=make_profile(tmp_path)
    worker=json.loads(Path(profile['owner']['path']).read_text())['workers'][0]
    worker['expected']['nativeStates']=['reconstructed'];worker['context_window']=16384
    worker['limits'].update(max_turns=10,deadline_ms=170000,max_session_tokens=10*(16384+512))
    validate_entry(worker,development=True)
    with pytest.raises(ValueError,match='ROOM_LIMIT'):validate_entry(worker)
    worker['expected']['nativeStates']=['retained']
    with pytest.raises(ValueError,match='ROOM_LIMIT'):validate_entry(worker,development=True)
    worker['expected']['nativeStates']=['reconstructed'];worker['limits']['max_session_tokens']+=1
    with pytest.raises(ValueError,match='ROOM_LIMIT'):validate_entry(worker,development=True)
