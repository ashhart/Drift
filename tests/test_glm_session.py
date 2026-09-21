"""Exercise native chat tool calls and explicit transcript reconstruction across turns."""
import copy

import pytest

from drift.serving.glm_session import GlmSession


class Transport:
    def __init__(self, turns):
        self.turns, self.bodies, self.closed = list(turns), [], False

    def stream(self, body, timeout):
        self.bodies.append(copy.deepcopy(body))
        yield from self.turns.pop(0)

    def count_tokens(self, body, timeout):
        return next((event['usage']['prompt_tokens'] for event in self.turns[0] if 'usage' in event), 20)

    def cancel(self):
        self.closed = True

    def close(self):
        self.closed = True


def turn(*deltas, reason='stop', input_tokens=20, output_tokens=4):
    return [*({'choices': [{'delta': delta, 'finish_reason': None}]} for delta in deltas),
            {'choices': [{'delta': {}, 'finish_reason': reason}]},
            {'choices': [], 'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens}}]


def opened(transport, prepare=None):
    backend = GlmSession(transport, model='synthetic-model', prepare_turn=prepare)
    backend.open({'limits': {'max_input_bytes': 4096, 'max_session_tokens': 1000, 'deadline_ms': 5000},
                  'system_prompt': ['own system'],
                  'tools': [{'name': 'read', 'description': 'Read own file', 'parameters': {'type': 'object'}}]})
    backend.own_prompt({'text': 'Build an API'})
    return backend


def test_native_tool_fragments_are_reconstructed_once_with_counted_prefill():
    stream = turn({'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                                  'function': {'name': 'read', 'arguments': '{"path":'}}]},
                  {'tool_calls': [{'index': 0, 'function': {'arguments': '"app.py"}'}}]}, reason='tool_calls')
    transport = Transport([stream, turn({'content': 'Done'}, input_tokens=50)])
    backend = opened(transport)
    first = list(backend.stream(32))
    assert first[0] == {'op': 'tool_call', 'payload': {'call_id': 'call-1', 'name': 'read', 'arguments': {'path': 'app.py'}}}
    backend.tool_result({'call_id': 'call-1', 'text': 'own file contents', 'is_error': False})
    second = list(backend.stream(32))
    messages = transport.bodies[1]['messages']
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'tool']
    assert messages[-1]['tool_call_id'] == 'call-1'
    assert second[-1]['payload']['usage']['input_tokens'] == 50
    assert backend.capabilities['native_state'] == 'reconstructed'


@pytest.mark.parametrize('stream', [turn({'tool_calls': [{'index': 0, 'id': 'x', 'function': {'name': 'unknown', 'arguments': '{}'}}]}, reason='tool_calls'),
                                  turn({'tool_calls': [{'index': 0, 'id': 'x', 'function': {'name': 'read', 'arguments': '{broken'}}]}, reason='tool_calls'),
                                  turn({'content': 'incomplete'})[:-1]])
def test_unknown_tool_invalid_json_and_missing_usage_poison_session(stream):
    backend = opened(Transport([stream]))
    with pytest.raises(ValueError):
        list(backend.stream(32))
    with pytest.raises(ValueError):
        backend.own_prompt({'text': 'continue'})


def test_every_reconstructed_turn_requires_memory_preparation_again():
    prepared = []
    def prepare(body):
        prepared.append(copy.deepcopy(body['messages']))
        return {'kv_transfer_params': {'drift_session': 'synthetic-' + str(len(prepared))}}
    transport = Transport([turn({'content': 'one'}), turn({'content': 'two'})])
    backend = opened(transport, prepare)
    list(backend.stream(32))
    backend.own_prompt({'text': 'continue'})
    list(backend.stream(32))
    assert len(prepared) == 2
    assert transport.bodies[0]['kv_transfer_params'] != transport.bodies[1]['kv_transfer_params']
    assert prepared[1][-2:] == [{'role': 'assistant', 'content': 'one'}, {'role': 'user', 'content': 'continue'}]


def test_memory_hook_cannot_replace_own_messages_or_tools():
    backend = opened(Transport([]), lambda body: {'messages': [{'role': 'user', 'content': 'foreign text'}]})
    with pytest.raises(ValueError):
        list(backend.stream(32))


def test_cancel_waits_for_server_recovery_evidence_before_returning():
    transport = Transport([])
    events = []
    def recovered():
        assert transport.closed
        events.append('idle')
    backend = GlmSession(transport, model='synthetic-model', recovery=recovered)
    backend.cancel(1)
    assert events == ['idle']


def test_unqualified_cancel_is_not_advertised():
    backend = opened(Transport([]))
    assert not backend.capabilities['cancellation']
    with pytest.raises(ValueError):
        backend.cancel(1)


def test_control_update_preserves_prior_history_and_waits_for_tool_group():
    stream = turn({'tool_calls': [{'index': 0, 'id': 'call-1', 'function': {'name': 'read', 'arguments': '{}'}}]}, reason='tool_calls')
    transport = Transport([stream, turn({'content': 'done'}, input_tokens=70)])
    backend = opened(transport)
    list(backend.stream(32))
    before = copy.deepcopy(backend.messages)
    backend.own_control({'system_prompt': ['own todo update']})
    assert backend.messages == before
    backend.tool_result({'call_id': 'call-1', 'text': 'own tool result', 'is_error': False})
    result = list(backend.stream(32))
    messages = transport.bodies[-1]['messages']
    assert messages[:len(before)] == before
    assert [m['role'] for m in messages[-2:]] == ['tool', 'system']
    assert messages[-1]['content'].endswith('own todo update')
    assert result[-1]['payload']['usage']['input_tokens'] == 70
