import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/omp'))


def entry(state, maximum):
    from native_room_profile import LIMITS
    return dict(identity=dict(session='fresh', worker='glm', model_id='glm', model_sha256='a'*64, translator_sha256='b'*64),
                expected=dict(backend='live', nativeStates=[state]), experimental_multi_turn=True,
                memory_mode='no-link', limits={**LIMITS, 'max_session_tokens': maximum}, context_window=16384)


def test_explicit_reconstructed_prefill_budget_preserves_other_limits():
    from native_room_profile import validate_entry
    validate_entry(entry('reconstructed', 65536))
    validate_entry(entry('reconstructed', 20000))
    validate_entry(entry('retained', 20000))
    for value in [entry('retained', 65536), entry('reconstructed', 65537), entry('reconstructed', True)]:
        with pytest.raises(ValueError): validate_entry(value)
    value = entry('reconstructed', 65536)
    value['limits']['deadline_ms'] = 60000
    with pytest.raises(ValueError): validate_entry(value)


def test_room_result_uses_each_frozen_worker_budget():
    from native_room_evidence import room_ok
    facts = dict(errors=[], blocked=0, task_calls=1, hub_calls=1, todo_calls=1, workers={
        'glm': dict(turns=4, input_tokens=40000, output_tokens=100, shutdown=True),
        'qwen': dict(turns=3, input_tokens=5000, output_tokens=100, shutdown=True)})
    assert not room_ok(facts, ['glm', 'qwen'])
    assert room_ok(facts, ['glm', 'qwen'], {'glm': 65536, 'qwen': 20000})
    facts['workers']['glm']['input_tokens'] = 65536
    assert not room_ok(facts, ['glm', 'qwen'], {'glm': 65536, 'qwen': 20000})
