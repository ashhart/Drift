import json
from types import SimpleNamespace
import pytest
from drift.serving.worker_session import WorkerSession
from drift.serving.worker_studio_diagnostics import instrument_studio
from test_worker_session import Backend, LIMITS, PINS, command


def private_path(tmp_path):
    root = tmp_path/'private'; root.mkdir(mode=0o700)
    return root/'receipt.json'


def studio(path):
    class Native(Backend):
        input_tokens = total_tokens = 0
        tokenizer = SimpleNamespace(decode=lambda *args: '')
        parse_tools = staticmethod(lambda *args: ('', []))
        handle = staticmethod(lambda command: {})
        def close(self):
            self.receipt_present_before_close = bool(path.read_bytes())
            super().close()
    return instrument_studio(Native(), path)


def glm(path):
    from drift.serving.glm_session import GlmSession
    from drift.serving.glm_diagnostics import GlmDiagnostics
    from test_glm_session import Transport
    return GlmSession(Transport([]), model='fixture', diagnostics=GlmDiagnostics(path))


@pytest.mark.parametrize('factory', [studio, glm])
def test_protocol_rejection_before_backend_dispatch_writes_private_receipt(tmp_path, factory):
    path = private_path(tmp_path); backend = factory(path)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    assert list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': ['NEVER_EXPORT'], 'tools': []})))[0]['op'] == 'opened'
    assert list(session.handle(command(1, 'own_prompt', {'text': 'NEVER_EXPORT_FIRST'})))[0]['op'] == 'own_prompt_ack'
    reply = list(session.handle(command(2, 'own_prompt', {'text': 'NEVER_EXPORT_SECOND'})))
    assert reply[0]['payload'] == {'code': 'PROTOCOL'}
    report = json.loads(path.read_text())
    assert report['phase'] == 'protocol_operation'
    assert report['counts']['protocol_error'] == 1
    assert report['counts']['protocol_op'] == 2
    assert report['counts']['protocol_seq'] == 3
    assert report['counts']['protocol_accepted_seq'] == 3
    assert report['counts']['protocol_ready'] == 1
    assert report['counts']['protocol_pending'] == report['counts']['used_tokens'] == report['counts']['protocol_turns'] == 0
    assert 'NEVER_EXPORT' not in path.read_text() and 'text' not in report['counts']
    assert path.stat().st_mode & 0o777 == 0o600
    if factory is studio:
        assert backend.backend_impl.receipt_present_before_close


def test_malformed_envelope_never_exports_untrusted_opcode_sequence_or_paths(tmp_path):
    path = private_path(tmp_path); backend = studio(path)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    response = list(session.handle({'op': 'NEVER_EXPORT_OPCODE', 'seq': 'NEVER_EXPORT_SEQ', 'payload': {'private': '/NEVER_EXPORT_PATH'}}))
    assert response[0]['payload'] == {'code': 'PROTOCOL'}
    report = json.loads(path.read_text())
    assert report['phase'] == 'protocol_admit'
    assert report['counts']['protocol_op'] == report['counts']['protocol_seq'] == 0
    assert 'NEVER_EXPORT' not in path.read_text()
    assert all(frame['file'] in ('worker_session.py', 'worker_contract.py', 'external') for error in report['errors'] for frame in error['frames'])


def test_first_backend_failure_remains_the_owner_receipt(tmp_path):
    path = private_path(tmp_path); backend = studio(path)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    backend.diagnostics.record('native_generate', ValueError('NEVER_EXPORT'), {'generated_tokens': 2})
    before = path.read_bytes()
    list(session.handle({'op': 'NEVER_EXPORT'}))
    assert path.read_bytes() == before


def test_protocol_receipt_counts_pending_and_budget_without_exporting_call_id(tmp_path):
    path = private_path(tmp_path); backend = studio(path)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})))
    session.tokens, session.turns = 7, 1
    session.pending.add('NEVER_EXPORT_PENDING')
    response = list(session.handle(command(1, 'tool_result', {'call_id': 'NEVER_EXPORT_WRONG', 'text': 'NEVER_EXPORT_TEXT', 'is_error': False})))
    assert response[0]['payload'] == {'code': 'PROTOCOL'}
    report = json.loads(path.read_text())
    assert report['counts']['protocol_op'] == 5 and report['counts']['protocol_pending'] == 1
    assert report['counts']['used_tokens'] == 7 and report['counts']['budget_tokens'] == LIMITS['max_session_tokens']
    assert report['counts']['protocol_turns'] == 1 and 'NEVER_EXPORT' not in path.read_text()


def test_broken_diagnostic_hook_never_replaces_protocol_error_or_cleanup(tmp_path):
    path = private_path(tmp_path); backend = studio(path)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})))
    def broken(*args): raise OSError('NEVER_EXPORT_DIAGNOSTIC_FAILURE')
    backend.protocol_failure = broken
    response = list(session.handle(command(0, 'own_prompt', {'text': 'NEVER_EXPORT'})))
    assert response[0]['payload'] == {'code': 'PROTOCOL'}
    assert backend.backend_impl.calls[-1] == 'close'
