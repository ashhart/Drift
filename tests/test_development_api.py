import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts/omp'))


def test_activation_supplement_is_blocked_before_loading_or_dispatch(monkeypatch):
    module = importlib.import_module('development_api')
    monkeypatch.setattr(module, 'load', lambda *args: pytest.fail('must not load unqualified linked profile'))
    with pytest.raises(ValueError, match='LINKED_UNQUALIFIED'): module.run(SimpleNamespace(arm='activation_supplement'))


def test_public_task_separates_owner_sources_and_pins_the_final_verifier(tmp_path):
    from test_native_room import make_profile
    from native_room_profile import digest
    from native_room_stage import verify_stage
    from development_stage import prepare
    _path, profile = make_profile(tmp_path)
    evidence = tmp_path/'evidence'
    prepared = prepare(profile, evidence, Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'), 49274)
    task = Path(prepared['cwd']); scope_path = Path(prepared['env']['DRIFT_TASK_CONFIG'])
    scope = json.loads(scope_path.read_text())
    assert not scope_path.is_relative_to(task)
    assert scope['edit_paths'] == ['api.py'] and scope['read_only_paths'] == ['verify.py']
    assert scope['runs'][0]['artifacts'][0]['sha256'] == digest(task/'verify.py')
    assert '("127.0.0.1", 49274)' in (task/'verify.py').read_text()
    assert all(not Path(path).is_relative_to(task) for path in prepared['source_pins'])
    assert prepared['channels'] == ['peer-text', 'shared-public-artifact']
    verify_stage(prepared)
    (task/'verify.py').write_text('tampered')
    with pytest.raises(ValueError, match='STAGE_PIN'): verify_stage(prepared)


def test_development_verdict_uses_frozen_reconstructed_token_allowance(tmp_path, monkeypatch):
    from test_native_room import make_profile
    from native_room_profile import digest
    import development_api
    path, profile = make_profile(tmp_path)
    owner_path = Path(profile['owner']['path'])
    owner = json.loads(owner_path.read_text())
    owner['workers'][0]['limits']['max_session_tokens'] = 65536
    owner_path.write_text(json.dumps(owner)); profile['owner']['sha256'] = digest(owner_path)
    path.write_text(json.dumps(profile))
    args = SimpleNamespace(arm='text_only', profile=path, profile_sha256=digest(path), evidence=tmp_path/'trial', run=False,
                           python=Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'), port=49275)
    prepared = development_api.run(args)
    args.run = True; args.prepared_sha256 = prepared['prepared_sha256']
    def execute(command, cwd, env):
        facts = dict(errors=[], blocked=0, task_calls=1, hub_calls=1, todo_calls=1, public_verify_passes=1,
                     last_verified_api_sha256=digest(Path(cwd)/'api.py'), workers={
                         'drift-experimental/glm': dict(turns=4, input_tokens=40000, output_tokens=100, shutdown=True),
                         'drift-experimental/qwen': dict(turns=3, input_tokens=5000, output_tokens=100, shutdown=True)})
        (args.evidence/'facts.json').write_text(json.dumps(facts))
        return dict(exit_code=0, output_joined=True)
    monkeypatch.setattr(development_api, 'execute', execute)
    monkeypatch.setattr(development_api, 'read_cleanup', lambda item, fresh: None if fresh else {'passed': True})
    assert development_api.run(args)['verdict'] == 'PASSED'
