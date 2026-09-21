import json
from types import SimpleNamespace
import pytest
from drift.serving.worker_diagnostics import PrivateDiagnostics
from drift.serving.worker_studio_diagnostics import instrument_studio


def private_path(tmp_path):
    root = tmp_path / 'private'; root.mkdir(mode=0o700)
    return root / 'diagnostic.json'


def test_diagnostics_scrubs_exception_payload_chains_and_untrusted_frame_names(tmp_path):
    path = private_path(tmp_path)
    owner = PrivateDiagnostics(path)
    secret = 'private_secret_12345'
    namespace = {}
    exec(compile('def private_secret_12345():\n raise ValueError("private_secret_12345")\n', '/private_secret_12345.py', 'exec'), namespace)
    try:
        try: namespace[secret]()
        except Exception: raise RuntimeError(secret) from None
    except Exception as error:
        owner.record('native_generate', error, {'generation_attempts': 3, 'generated_tokens': 2})
    raw = path.read_text()
    assert secret not in raw
    report = json.loads(raw)
    assert report['phase'] == 'native_generate'
    assert [item['exception'] for item in report['errors']] == ['RuntimeError', 'ValueError']
    assert report['counts']['generated_tokens'] == 2
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError): PrivateDiagnostics(path)


def test_diagnostics_refuses_nonprivate_and_symlink_paths(tmp_path):
    public = tmp_path / 'public'; public.mkdir(mode=0o755)
    with pytest.raises(ValueError): PrivateDiagnostics(public / 'out.json')
    private = tmp_path / 'private'; private.mkdir(mode=0o700)
    alias = tmp_path / 'alias'; alias.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError): PrivateDiagnostics(alias / 'out.json')


def test_generation_failure_is_owner_only_and_preserves_original_error(tmp_path):
    path = private_path(tmp_path)
    class Backend:
        backend = 'live'
        capabilities = {'native_state': 'retained'}
        input_tokens = 7
        total_tokens = 0
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: 'private text')
        parse_tools = staticmethod(lambda *args: ('private text', None))
        def handle(self, command): raise ValueError('private token IDs and tensor values')
        def stream(self, maximum):
            self.handle({'op': 'generate_own', 'tokens': 1})
            yield {'op': 'text', 'payload': {'text': 'must not be emitted'}}
    observed = instrument_studio(Backend(), path)
    with pytest.raises(ValueError, match='private token IDs'):
        list(observed.stream(2))
    report = json.loads(path.read_text())
    assert report['phase'] == 'native_generate'
    assert report['counts']['generation_attempts'] == 1 and report['counts']['input_tokens'] == 7
    assert 'private token IDs' not in path.read_text()


def test_parse_and_serialize_phases_preserve_counts_without_text(tmp_path):
    path = private_path(tmp_path)
    class Backend:
        backend = 'live'; capabilities = {}; input_tokens = 3; total_tokens = 0
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: 'secret output')
        handle = staticmethod(lambda frame: {'ids': [17], 'done': False})
        parse_tools = staticmethod(lambda *args: ('secret output', None))
        def stream(self, maximum):
            result = self.handle({'op': 'generate_own'})
            text = self.tokenizer.decode(result['ids'])
            self.parse_tools(text)
            raise KeyError('secret key')
            yield
    observed = instrument_studio(Backend(), path)
    with pytest.raises(KeyError): list(observed.stream(1))
    report = json.loads(path.read_text())
    assert report['phase'] == 'stream_serialize' and report['counts']['generated_tokens'] == 1
    assert 'secret' not in path.read_text()


def test_worker_protocol_keeps_only_original_error_code(tmp_path):
    from drift.serving.worker_session import WorkerSession
    from drift.serving.worker_studio_backend import StudioBackend
    from test_worker_session import LIMITS, PINS, command
    path = private_path(tmp_path)
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2], decode=lambda *a, **kw: '')
    def handle(frame):
        if frame['op'] == 'generate_own':
            raise RuntimeError('secret native failure')
        return {}
    backend = instrument_studio(StudioBackend(handle, {}, tokenizer, lambda *a: ('', None), 0), path)
    session = WorkerSession('qwen', PINS, LIMITS, backend)
    assert list(session.handle(command(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})))[0]['op'] == 'opened'
    assert list(session.handle(command(1, 'own_prompt', {'text': 'private user text'})))[0]['op'] == 'own_prompt_ack'
    response = list(session.handle(command(2, 'stream', {'max_tokens': 2})))
    assert len(response) == 1 and response[0]['payload'] == {'code': 'WORKER'}
    assert 'private' not in json.dumps(response) and 'secret' not in path.read_text()
