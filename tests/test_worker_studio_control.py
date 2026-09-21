import pytest
from drift.serving.worker_studio_backend import StudioBackend
from test_worker_session import LIMITS, PINS


def test_explicit_qwen_control_format_is_lower_authority_user_input():
    backend = StudioBackend(lambda value: None, {}, None, None, control_format='qwen-reader-user-v1')
    backend.own_control({'system_prompt': ['initial snapshot', 'declared peer text']})
    assert backend.pending_control[0]['role'] == 'user'
    assert 'does not replace or outrank' in backend.pending_control[0]['content']
    assert backend.pending_control[0]['content'].endswith('initial snapshot\n\ndeclared peer text')


def test_unknown_model_control_format_fails_closed():
    with pytest.raises(RuntimeError, match='CAPABILITY'):
        StudioBackend(lambda value: None, {}, None, None, control_format='unknown')


def test_control_format_selection_is_checkpoint_specific(tmp_path):
    from drift.serving.worker_studio_control import checkpoint_control_format
    path = tmp_path / 'config.json'
    path.write_text('{"model_type":"qwen4_exp"}')
    assert checkpoint_control_format(tmp_path) == 'qwen-reader-user-v1'
    path.write_text('{"model_type":"unknown"}')
    assert checkpoint_control_format(tmp_path) is None
