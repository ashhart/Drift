"""Write one bounded owner-only failure receipt without private payloads or source lines."""
import json
import os
from pathlib import Path
import threading

PHASES = {'protocol_admit', 'protocol_operation', 'protocol_stream', 'open', 'own_prompt', 'tool_result', 'stream', 'native_open', 'native_prefill',
          'native_generate', 'stream_collect', 'decode', 'tool_parse', 'stream_serialize', 'close', 'glm_prepare', 'glm_count', 'glm_budget', 'glm_admit',
          'glm_dispatch', 'glm_receive', 'glm_parse', 'glm_usage'}
CLASSES = {'Exception', 'RuntimeError', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
           'AttributeError', 'MemoryError', 'TimeoutError', 'OSError', 'WorkerError', 'TemplateError', 'JSONDecodeError'}
FILES = {'worker_session.py', 'worker_own_control.py', 'worker_studio_backend.py', 'worker_studio.py', 'worker_native_gate.py', 'studio_drift_worker.py',
         'worker_codec.py', 'worker_contract.py', 'language.py', 'cache.py', 'utils.py', 'base.py',
         'tool_calling.py', 'qwen3_coder.py', 'tokenization_utils_base.py', 'tokenization_utils_fast.py', 'glm_session.py', 'glm_factory.py', 'glm_http.py', 'glm_http_input.py',
         'glm_chat_events.py', 'glm_guard.py', 'glm_diagnostics.py'}
FUNCTIONS = {'_admit', '_operation', '_deadline', 'apply_own_control', 'fields', 'integer', 'validate_open', 'stream', 'generate_native', 'continue_native', '__call__', 'run', 'handle', 'decode',
             'parse_tool_calls', '_parse_tool_calls_impl', 'parse_tool_call', 'assistant', 'append_many',
             'require', 'transaction', 'settle_native', 'update_and_fetch', 'update_indexer', 'forward', 'count_tokens', '_spawn', '_frames', 'admit', 'accept', 'finish',
             '_call', 'events', '_check', '_remaining'}
COUNTS = {'protocol_error', 'protocol_op', 'protocol_seq', 'protocol_accepted_seq', 'protocol_turns', 'protocol_pending',
          'protocol_ready', 'protocol_control_ready', 'protocol_active', 'protocol_control_updates', 'protocol_started', 'input_tokens', 'total_tokens', 'generation_attempts', 'generated_tokens', 'counted_tokens', 'admitted_tokens',
          'used_tokens', 'budget_tokens', 'request_dispatched', 'requested_output_tokens', 'tool_calls',
          'argument_bytes', 'finish_reason_code', 'json_error_position', 'json_document_chars'}


def sanitized_error(error):
    reports, seen = [], set()
    while error is not None and len(reports) < 3 and id(error) not in seen:
        seen.add(id(error)); frames = []; trace = error.__traceback__
        while trace is not None:
            code = trace.tb_frame.f_code
            filename = Path(code.co_filename).name
            frames.append({'file': filename if filename in FILES else 'external',
                           'function': code.co_name if code.co_name in FUNCTIONS else 'other',
                           'line': max(0, min(trace.tb_lineno, 1000000))})
            frames = frames[-16:]; trace = trace.tb_next
        name = type(error).__name__
        reports.append({'exception': name if name in CLASSES else 'Exception', 'frames': frames})
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return reports


class PrivateDiagnostics:
    def __init__(self, path):
        path = Path(path)
        parent = path.parent
        if (not path.is_absolute() or path.absolute() != path.resolve() or path.exists()
                or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o077
                or any((folder / '.git').exists() for folder in path.parents)):
            raise ValueError('INVALID_DIAGNOSTIC_PATH')
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.lock = threading.Lock()

    def record(self, phase, error, counts):
        if phase not in PHASES or set(counts) - COUNTS or any(type(n) is not int or not 0 <= n <= 2**31 for n in counts.values()):
            raise ValueError('INVALID_DIAGNOSTIC_FIELDS')
        report = {'phase': phase, 'errors': sanitized_error(error), 'counts': dict(counts)}
        raw = json.dumps(report, allow_nan=False).encode() + b'\n'
        if len(raw) > 16384:
            raise ValueError('DIAGNOSTIC_LIMIT')
        with self.lock:
            if self.fd is None:
                return
            try:
                if os.write(self.fd, raw) != len(raw):
                    raise OSError('DIAGNOSTIC_WRITE_FAILED')
            finally:
                os.close(self.fd); self.fd = None

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd); self.fd = None


def protocol_failure(session, frame, phase, error):
    from drift.serving.worker_contract import WorkerError
    hook = getattr(session.backend, 'protocol_failure', None)
    if not callable(hook): return
    operations = {'open': 1, 'own_prompt': 2, 'own_control': 3, 'own_control_append': 4,
                  'tool_result': 5, 'stream': 6, 'cancel': 7, 'close': 8}
    frame = frame if type(frame) is dict else {}
    operation = frame.get('op'); sequence = frame.get('seq')
    code = str(error) if isinstance(error, WorkerError) else 'WORKER'
    counts = {'protocol_error': {'PROTOCOL': 1, 'LIMIT': 2, 'CAPABILITY': 3, 'WORKER': 4}.get(code, 4),
              'protocol_op': operations.get(operation, 0) if type(operation) is str else 0,
              'protocol_seq': sequence if type(sequence) is int and 0 <= sequence <= 2**31 else 0,
              'protocol_accepted_seq': min(session.seq, 2**31), 'protocol_turns': session.turns,
              'used_tokens': session.tokens, 'budget_tokens': session.limits['max_session_tokens'],
              'protocol_pending': len(session.pending), 'protocol_ready': int(session.ready),
              'protocol_control_ready': int(session.control_ready), 'protocol_active': int(session.active_seq is not None),
              'protocol_control_updates': session.control_updates, 'protocol_started': int(session.started is not None)}
    hook(phase, error, counts)
