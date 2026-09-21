from copy import deepcopy
import pytest
from drift.serving.worker_studio_backend import StudioBackend
from test_worker_own_control import setup
from test_worker_session import Backend, LIMITS, PINS, command
from test_glm_session import Transport, opened, turn


def test_control_only_wake_consumes_one_pending_control_without_user_replay():
    class Controlled(Backend):
        def own_control(self, payload): self.calls.append(deepcopy(payload))
    backend = Controlled(); session = setup(backend, True)
    list(session.handle(command(1, 'own_prompt', {'text': 'first'})))
    list(session.handle(command(2, 'stream', {'max_tokens': 2})))
    list(session.handle(command(3, 'own_control', {'system_prompt': ['peer finished']})))
    result = list(session.handle(command(4, 'stream', {'max_tokens': 2})))
    assert result[-1]['op'] == 'terminal'
    assert session.tokens == 8 and session.turns == 2
    assert sum('text' in call for call in backend.calls) == 1
    assert list(session.handle(command(5, 'stream', {'max_tokens': 2})))[0]['payload']['code'] == 'PROTOCOL'


def test_glm_control_only_wake_preserves_history_and_counts_reconstructed_prefill():
    transport = Transport([turn({'content': 'first'}), turn({'content': 'second'}, input_tokens=70)])
    backend = opened(transport)
    list(backend.stream(8)); previous = deepcopy(backend.messages)
    backend.own_control({'system_prompt': ['peer finished']})
    result = list(backend.stream(8))
    assert transport.bodies[-1]['messages'][:len(previous)] == previous
    assert transport.bodies[-1]['messages'][-1]['role'] == 'system'
    assert transport.bodies[-1]['messages'][-1]['content'].endswith('peer finished')
    assert backend.pending_control == [] and backend.used == 98
    assert result[-1]['payload']['usage']['input_tokens'] == 70


@pytest.mark.parametrize('rewrite', [False, True])
def test_studio_control_only_wake_appends_exact_native_suffix_and_preserves_cache(rewrite):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): return [99, 4, 5] if rewrite else [1, 2, 7, 9, 4, 5]
        def decode(self, ids, **kwargs): return 'done'
    cache = object(); state = {'cache': cache}; operations = []
    def handle(value):
        operations.append(value)
        return {'ids': [8], 'stop_id': None, 'done': True}
    backend = StudioBackend(handle, state, Tokenizer(), lambda *args: ('done', []))
    backend.open({**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})
    backend.codec.messages = [{'role': 'assistant', 'content': 'first'}]
    backend.codec.consumed = [1, 2, 7, 9]
    backend.own_control({'system_prompt': ['peer finished']})
    if rewrite:
        with pytest.raises(RuntimeError, match='CAPABILITY'): list(backend.stream(2))
        assert len(operations) == 1
    else:
        result = list(backend.stream(2))
        assert operations[1] == {'op': 'continue', 'ids': [4, 5]}
        assert result[-1]['payload']['usage']['input_tokens'] == 2
        assert backend.pending_control == []
    assert state['cache'] is cache
