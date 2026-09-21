from test_worker_session import Backend, LIMITS, PINS
from drift.serving.worker_session import WorkerSession


def test_actual_protocol_accepts_provider_first_sequence_one():
    frame = {'v': 1, 'session': 's1', 'worker': 'qwen', 'seq': 1, 'op': 'open', 'payload': {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []}}
    session = WorkerSession('qwen', PINS, LIMITS, Backend())
    assert list(session.handle(frame))[0]['op'] == 'opened'
