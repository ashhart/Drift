from copy import deepcopy
from drift.serving.worker_session import WorkerSession
from drift.serving.worker_studio_backend import StudioBackend
from test_worker_session import Backend, LIMITS, PINS, command


def setup(backend, enabled):
    session = WorkerSession('qwen', PINS, LIMITS, backend, allow_own_control=enabled)
    result = list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': ['original'], 'tools': []})))
    assert result[0]['op'] == 'opened'
    return session


def test_own_control_requires_explicit_opt_in_and_backend_support():
    session = setup(Backend(), False)
    reply = list(session.handle(command(1, 'own_control', {'system_prompt': ['update']})))
    assert reply[0]['payload']['code'] == 'CAPABILITY'
    session = setup(Backend(), True)
    assert list(session.handle(command(1, 'own_control', {'system_prompt': ['update']})))[0]['payload']['code'] == 'CAPABILITY'


def test_own_control_is_bounded_and_preserves_completed_turn_history():
    class Controlled(Backend):
        def own_control(self, payload): self.calls.append(deepcopy(payload))
    backend = Controlled(); session = setup(backend, True)
    list(session.handle(command(1, 'own_prompt', {'text': 'one'})))
    list(session.handle(command(2, 'stream', {'max_tokens': 2})))
    assert list(session.handle(command(3, 'own_control', {'system_prompt': ['new own todo']})))[0]['op'] == 'own_control_ack'
    assert session.tokens == 4 and session.turns == 1
    list(session.handle(command(4, 'own_prompt', {'text': 'two'})))
    list(session.handle(command(5, 'stream', {'max_tokens': 2})))
    assert session.tokens == 8 and session.turns == 2
    assert {'text': 'one'} in backend.calls and {'text': 'two'} in backend.calls
    assert {'system_prompt': ['new own todo']} in backend.calls


def test_control_cannot_change_an_already_prefilled_turn():
    class Controlled(Backend):
        def own_control(self, payload): raise AssertionError('must not dispatch')
    session = setup(Controlled(), True)
    list(session.handle(command(1, 'own_prompt', {'text': 'one'})))
    assert list(session.handle(command(2, 'own_control', {'system_prompt': ['changed']})))[0]['payload']['code'] == 'PROTOCOL'


def test_studio_stages_control_as_new_own_input_without_cache_reset():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages) == 2 else [1, 2, 7, 9, 4, 5]
    state = {'cache': object()}; cache = state['cache']; operations = []
    backend = StudioBackend(lambda value: operations.append(value), state, Tokenizer(), lambda *args: ('', []))
    backend.open({**PINS, 'limits': LIMITS, 'system_prompt': ['original'], 'tools': []})
    backend.own_prompt({'text': 'first'})
    backend.codec.assistant({'role': 'assistant', 'content': 'done'}, [7], 9)
    past = deepcopy(backend.codec.messages)
    backend.own_control({'system_prompt': ['new own todo']})
    assert backend.codec.messages == past
    assert len(operations) == 2
    backend.own_prompt({'text': 'next'})
    assert backend.codec.messages[:len(past)] == past
    assert backend.codec.messages[-2]['role'] == 'system'
    assert 'new own todo' in backend.codec.messages[-2]['content']
    assert state['cache'] is cache and operations[-1] == {'op': 'continue', 'ids': [4, 5]}


def test_control_updates_share_session_bounds_and_reject_malformed_payloads():
    class Controlled(Backend):
        def own_control(self, payload): self.calls.append(payload)
    backend = Controlled(); session = setup(backend, True)
    for index in range(LIMITS['max_turns']):
        assert list(session.handle(command(index + 1, 'own_control', {'system_prompt': [str(index)]})))[0]['op'] == 'own_control_ack'
    assert list(session.handle(command(4, 'own_control', {'system_prompt': ['too many']})))[0]['payload']['code'] == 'LIMIT'
    other = setup(Controlled(), True)
    assert list(other.handle(command(1, 'own_control', {'system_prompt': 'not a list'})))[0]['payload']['code'] == 'PROTOCOL'


def test_studio_control_waits_until_entire_tool_result_group_is_present():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): return [1, 2, 7, 9, 4, 5]
    backend = StudioBackend(lambda value: None, {'cache': object()}, Tokenizer(), lambda *args: ('', []))
    backend.open({**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})
    backend.codec.messages = [{'role': 'assistant', 'content': 'two tool calls'}]
    backend.codec.consumed = [1, 2, 7, 9]
    backend.awaiting_tools = 2
    backend.own_control({'system_prompt': ['updated own todo']})
    backend.tool_result({'call_id': 'a', 'text': 'first', 'is_error': False})
    assert len(backend.codec.messages) == 1
    backend.tool_result({'call_id': 'b', 'text': 'second', 'is_error': False})
    assert [message['role'] for message in backend.codec.messages] == ['assistant', 'tool', 'tool', 'system']
    assert backend.input_tokens == 2 and backend.pending_control == []


def test_retained_template_rewrite_is_rejected_without_resetting_cache():
    import pytest
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): return [99, 4, 5]
    operations = []; cache = object(); state = {'cache': cache}
    backend = StudioBackend(lambda value: operations.append(value), state, Tokenizer(), lambda *args: ('', []))
    backend.open({**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})
    backend.codec.consumed = [1, 2]
    backend.own_control({'system_prompt': ['new']})
    with pytest.raises(RuntimeError, match='CAPABILITY'):
        backend.own_prompt({'text': 'next'})
    assert operations == [{'op': 'start', 'reserve': 4096}] and state['cache'] is cache
