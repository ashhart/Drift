import pytest
from drift.serving.worker_session import WorkerSession
from drift.serving.worker_studio import continue_native, generate_native

PINS = {'model_id': 'fixture', 'model_sha256': 'a' * 64, 'translator_sha256': 'b' * 64}
LIMITS = {'max_input_bytes': 1024, 'max_output_tokens': 8, 'max_session_tokens': 32, 'max_turns': 3, 'deadline_ms': 5000}


class Backend:
    backend = 'fixture'
    capabilities = {'native_state': 'fixture', 'tool_calls': True, 'cancellation': True}

    def __init__(self):
        self.calls = []

    def open(self, payload): self.calls.append('open')
    def own_prompt(self, payload): self.calls.append(payload)
    def tool_result(self, payload): self.calls.append(payload)
    def stream(self, max_tokens):
        yield {'op': 'text', 'payload': {'text': 'hello'}}
        yield {'op': 'terminal', 'payload': {'reason': 'stop', 'usage': {'input_tokens': 3, 'output_tokens': 1}}}
    def cancel(self, target_seq): self.calls.append('cancel')
    def close(self): self.calls.append('close')


def command(seq, op, payload):
    return {'v': 1, 'session': 's1', 'worker': 'qwen', 'seq': seq + 1, 'op': op, 'payload': payload}


def opened(backend=None):
    backend = backend or Backend()
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    replies = list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})))
    assert replies[0]['op'] == 'opened'
    return session, backend


def test_pins_fail_before_backend_mutation():
    backend = Backend()
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    reply = list(session.handle(command(0, 'open', {**PINS, 'model_sha256': 'c' * 64, 'limits': LIMITS, 'system_prompt': [], 'tools': []})))
    assert reply[0]['payload']['code'] == 'CAPABILITY'
    assert backend.calls == []


def test_own_prompt_stream_usage_and_duplicate_rejection():
    session, backend = opened()
    assert list(session.handle(command(1, 'own_prompt', {'text': 'private'})))[0]['op'] == 'own_prompt_ack'
    replies = list(session.handle(command(2, 'stream', {'max_tokens': 8})))
    assert [r['op'] for r in replies] == ['text', 'terminal']
    assert session.tokens == 4
    assert list(session.handle(command(2, 'own_prompt', {'text': 'repeat'})))[0]['op'] == 'error'
    assert backend.calls.count({'text': 'repeat'}) == 0


def test_oversize_and_cross_worker_rejected_without_input_delivery():
    session, backend = opened()
    bad = command(1, 'own_prompt', {'text': 'x' * 2000})
    assert list(session.handle(bad))[0]['payload']['code'] == 'LIMIT'
    assert backend.calls == ['open', 'close']
    other, backend = opened()
    bad = command(1, 'own_prompt', {'text': 'private'}); bad['worker'] = 'glm'
    assert list(other.handle(bad))[0]['payload']['code'] == 'PROTOCOL'


def test_backend_failure_is_sanitized_and_poisoned():
    class Failing(Backend):
        def own_prompt(self, payload): raise ValueError('private contents')
    session, backend = opened(Failing())
    reply = list(session.handle(command(1, 'own_prompt', {'text': 'private'})))
    assert reply[0]['payload'] == {'code': 'WORKER'}
    assert 'private' not in repr(reply)
    assert list(session.handle(command(2, 'stream', {'max_tokens': 1})))[0]['op'] == 'error'


def test_stream_requires_terminal_and_honest_usage():
    class Broken(Backend):
        def stream(self, max_tokens):
            yield {'op': 'terminal', 'payload': {'reason': 'stop', 'usage': {'input_tokens': 0, 'output_tokens': 99}}}
    session, _ = opened(Broken())
    list(session.handle(command(1, 'own_prompt', {'text': 'hi'})))
    assert list(session.handle(command(2, 'stream', {'max_tokens': 2})))[0]['payload']['code'] == 'LIMIT'


def test_pending_stop_is_consumed_before_continuation_and_cache_retained():
    cache = object(); state = {'cache': cache, 'done': True, 'pending_stop': 9, 'last': [0, 1]}
    consumed = []
    continue_native(state, [4, 5], lambda ids: consumed.extend(ids))
    assert consumed == [9, 4, 5]
    assert state['cache'] is cache and not state['done'] and state['pending_stop'] is None
    with pytest.raises(ValueError): continue_native(state, [], lambda ids: None)
    state['poisoned'] = True
    with pytest.raises(RuntimeError): continue_native(state, [7], lambda ids: None)


def test_generate_records_unconsumed_boundary_token():
    class Logits:
        def argmax(self): return 9
    state = {'cache': object(), 'last': Logits(), 'done': False}
    assert generate_native(state, 4, {9}, lambda ids: pytest.fail('stop must be deferred'), str) == {'text': '[]', 'tokens': 0, 'done': True}
    assert state['pending_stop'] == 9


def test_cancel_ack_waits_for_stream_exit():
    import threading
    class Waiting(Backend):
        def __init__(self):
            super().__init__(); self.running = threading.Event(); self.release = threading.Event()
        def stream(self, max_tokens):
            self.running.set(); self.release.wait(1)
            yield {'op': 'text', 'payload': {'text': 'must not escape after cancel'}}
        def cancel(self, target_seq): self.release.set()
    session, backend = opened(Waiting())
    list(session.handle(command(1, 'own_prompt', {'text': 'hi'})))
    replies = []
    thread = threading.Thread(target=lambda: replies.extend(session.handle(command(2, 'stream', {'max_tokens': 2}))))
    thread.start(); assert backend.running.wait(1)
    ack = list(session.handle(command(3, 'cancel', {'target_seq': 3})))
    thread.join(1)
    assert ack[0]['op'] == 'cancelled' and not thread.is_alive()
    assert replies == []


def test_checkpoint_codec_rejects_template_prefix_rewriting():
    from drift.serving.worker_codec import CheckpointCodec
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages) == 1 else [1, 2, 7, 9, 4, 5]
    codec = CheckpointCodec(Tokenizer(), [], [])
    assert codec.append({'role': 'user', 'content': 'one'}) == [1, 2]
    codec.assistant({'role': 'assistant', 'content': 'two'}, [7], 9)
    assert codec.append({'role': 'tool', 'content': 'three'}) == [4, 5]
    codec.consumed.append(99)
    with pytest.raises(RuntimeError, match='CAPABILITY'):
        codec.append({'role': 'user', 'content': 'changed'})


def test_stdio_runs_real_dispatch_and_rejects_long_lines():
    import io
    import json
    from drift.serving.worker_stdio import serve
    backend = Backend(); session = WorkerSession('qwen', PINS, LIMITS, backend)
    frames = [command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []}), command(1, 'close', {})]
    output = io.StringIO()
    assert serve(session, io.StringIO(''.join(json.dumps(x) + '\n' for x in frames)), output) == 0
    assert [json.loads(line)['op'] for line in output.getvalue().splitlines()] == ['opened', 'closed']
    session = WorkerSession('qwen', PINS, LIMITS, Backend())
    assert serve(session, io.StringIO('x' * 3000 + '\n'), io.StringIO()) == 2


def test_executable_stdio_cli_uses_explicit_backend_factory(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    (tmp_path / 'fixture_backend.py').write_text('from test_worker_session import Backend\ndef make(config): return Backend()\n')
    config = tmp_path / 'config.json'
    import hashlib
    artifact = tmp_path / 'artifact'; artifact.write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'weights_verification': 'fixture has no weights', 'artifacts': {'fixture': 'artifact'}, 'sha256': {'fixture': hashlib.sha256(b'fixture').hexdigest()}}))
    pins = {**PINS, 'model_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'translator_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    config.write_text(json.dumps({'worker': 'qwen', 'pins': pins, 'limits': LIMITS, 'model_manifest_path': str(manifest), 'translator_manifest_path': str(manifest)}))
    frames = [command(0, 'open', {**pins, 'limits': LIMITS, 'system_prompt': [], 'tools': []}), command(1, 'close', {})]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(tmp_path), str(Path.cwd()), str(Path.cwd() / 'tests')))}
    result = subprocess.run([sys.executable, '-m', 'drift.serving.worker_stdio', '--config', str(config), '--backend', 'fixture_backend:make'], input=''.join(json.dumps(x) + '\n' for x in frames), text=True, capture_output=True, env=env, timeout=5)
    assert result.returncode == 0 and not result.stderr
    assert [json.loads(line)['op'] for line in result.stdout.splitlines()] == ['opened', 'closed']


def test_manifest_digest_and_artifact_content_both_verified(tmp_path):
    import hashlib
    import json
    from drift.serving.worker_artifacts import verify_worker_manifests
    artifact = tmp_path / 'tokenizer'; artifact.write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'weights_verification': 'fixture has no weights', 'artifacts': {'tokenizer': 'tokenizer'}, 'sha256': {'tokenizer': hashlib.sha256(b'fixture').hexdigest()}}))
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = {'pins': {'model_sha256': checksum, 'translator_sha256': checksum}, 'model_manifest_path': str(manifest), 'translator_manifest_path': str(manifest)}
    assert verify_worker_manifests(config)['model']['verified_artifacts'] == 1
    artifact.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='CAPABILITY'): verify_worker_manifests(config)


def test_manifest_artifact_byte_cap_prevents_oversize_hashing(tmp_path):
    import hashlib
    import json
    from drift.serving.worker_artifacts import verify_worker_manifests
    artifact = tmp_path / 'artifact'; artifact.write_bytes(b'large')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'weights_verification': 'not hashed', 'artifacts': {'one': 'artifact'}, 'sha256': {'one': hashlib.sha256(b'large').hexdigest()}}))
    value = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = {'pins': {'model_sha256': value, 'translator_sha256': value}, 'model_manifest_path': str(manifest), 'translator_manifest_path': str(manifest), 'max_verified_artifact_bytes': 1}
    with pytest.raises(RuntimeError, match='LIMIT'): verify_worker_manifests(config)


def test_terminal_is_emitted_after_stream_is_no_longer_active():
    session, _ = opened()
    list(session.handle(command(1, 'own_prompt', {'text': 'hi'})))
    stream = session.handle(command(2, 'stream', {'max_tokens': 4}))
    assert next(stream)['op'] == 'text'
    assert next(stream)['op'] == 'terminal'
    assert session.active_seq is None
    assert list(session.handle(command(3, 'own_prompt', {'text': 'next'})))[0]['op'] == 'own_prompt_ack'
    list(stream)


def test_cancelled_backend_exception_does_not_precede_cancellation_ack():
    import threading
    class Interrupted(Backend):
        def __init__(self):
            super().__init__()
            self.started = threading.Event(); self.stopped = threading.Event()
        def stream(self, max_tokens):
            self.started.set(); self.stopped.wait(1)
            raise RuntimeError('interrupted private stream')
            yield
        def cancel(self, target_seq): self.stopped.set()
        def close(self): pass
    session, backend = opened(Interrupted())
    list(session.handle(command(1, 'own_prompt', {'text': 'hi'})))
    output = []
    thread = threading.Thread(target=lambda: output.extend(session.handle(command(2, 'stream', {'max_tokens': 4}))))
    thread.start(); assert backend.started.wait(1)
    reply = list(session.handle(command(3, 'cancel', {'target_seq': 3})))
    thread.join(1)
    assert output == [] and reply[0]['op'] == 'cancelled'


def test_stdio_admits_the_next_stream_while_the_last_turns_thread_finishes():
    import io
    import json
    import threading
    import time
    from drift.serving.worker_stdio import serve

    class Lingering(threading.Event):   # the turn's thread sets this last, after its terminal frame is written
        def set(self):
            if threading.current_thread() is not threading.main_thread():
                time.sleep(0.3)
            super().set()

    class Sink(io.StringIO):
        def __init__(self):
            super().__init__(); self.terminals = threading.Semaphore(0)

        def write(self, text):
            written = super().write(text)
            if json.loads(text)['op'] == 'terminal':
                self.terminals.release()
            return written

    class Client:   # sends the frame after a stream only once that stream's terminal frame is out, as a caller does
        def __init__(self, frames, sink):
            self.frames, self.sink, self.after_stream = list(frames), sink, False

        def readline(self, limit):
            if not self.frames:
                return ''
            if self.after_stream:
                assert self.sink.terminals.acquire(timeout=5)
            frame = self.frames.pop(0); self.after_stream = frame['op'] == 'stream'
            return json.dumps(frame) + '\n'

    session = WorkerSession('qwen', PINS, LIMITS, Backend()); session.stream_finished = Lingering()
    frames = [command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []}),
              command(1, 'own_prompt', {'text': 'a'}), command(2, 'stream', {'max_tokens': 8}),
              command(3, 'own_prompt', {'text': 'b'}), command(4, 'stream', {'max_tokens': 8}), command(5, 'close', {})]
    sink = Sink()
    assert serve(session, Client(frames, sink), sink) == 0
    assert [json.loads(line)['op'] for line in sink.getvalue().splitlines()] == [
        'opened', 'own_prompt_ack', 'text', 'terminal', 'own_prompt_ack', 'text', 'terminal', 'closed']


def test_stdio_still_refuses_a_stream_sent_during_a_live_turn():
    import io
    import json
    import threading
    from drift.serving.worker_stdio import serve
    release = threading.Event()

    class Slow(Backend):
        def stream(self, max_tokens):
            release.wait(5)
            yield from Backend.stream(self, max_tokens)

        def cancel(self, target_seq):
            release.set(); super().cancel(target_seq)

    session = WorkerSession('qwen', PINS, LIMITS, Slow())
    frames = [command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []}),
              command(1, 'own_prompt', {'text': 'a'}), command(2, 'stream', {'max_tokens': 8}), command(3, 'stream', {'max_tokens': 8})]
    output = io.StringIO()
    try:
        assert serve(session, io.StringIO(''.join(json.dumps(x) + '\n' for x in frames)), output) == 2
    finally:
        release.set()
    assert [json.loads(line)['op'] for line in output.getvalue().splitlines()][-1] == 'error'
