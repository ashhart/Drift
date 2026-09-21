from types import SimpleNamespace
from drift.serving.worker_studio_backend import StudioBackend
from test_worker_session import LIMITS, PINS, command
from drift.serving.worker_session import WorkerSession


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2] if len(messages) == 1 else [1, 2, 7, 9, 4, 5]
    def decode(self, ids, **kwargs): return 'synthetic tool call' if ids == [7] else ''


def test_concrete_studio_adapter_tool_result_reuses_native_handle():
    calls = []; cache = object(); state = {}; index = 0
    def handle(frame):
        nonlocal index
        calls.append(frame)
        if frame['op'] == 'start': state['cache'] = cache; return {}
        if frame['op'] == 'continue': return {'tokens': len(frame['ids'])}
        index += 1
        return {'ids': [7] if index % 2 else [], 'stop_id': None if index % 2 else 9, 'done': index % 2 == 0}
    call = SimpleNamespace(id='call1', function=SimpleNamespace(name='inspect', arguments='{}'), model_dump=lambda: {'id': 'call1', 'type': 'function', 'function': {'name': 'inspect', 'arguments': '{}'}})
    backend = StudioBackend(handle, state, Tokenizer(), lambda *args: ('', [call]), reserve=8)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    tools = [{'name': 'inspect', 'description': 'fixture', 'parameters': {'type': 'object'}}]
    assert list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': tools})))[0]['op'] == 'opened'
    list(session.handle(command(1, 'own_prompt', {'text': 'test'})))
    events = list(session.handle(command(2, 'stream', {'max_tokens': 4})))
    assert [event['op'] for event in events] == ['tool_call', 'terminal']
    assert events[-1]['payload']['usage'] == {'input_tokens': 2, 'output_tokens': 2}
    assert list(session.handle(command(3, 'tool_result', {'call_id': 'call1', 'text': 'ok', 'is_error': False})))[0]['op'] == 'tool_result_ack'
    assert state['cache'] is cache
    assert [frame['ids'] for frame in calls if frame['op'] == 'continue'] == [[1, 2], [4, 5]]
