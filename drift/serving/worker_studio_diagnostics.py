"""Instrument Studio phases without changing generation or private protocol messages."""
from drift.serving.worker_diagnostics import PrivateDiagnostics


class Handle:
    def __init__(self, original, observer):
        self.original, self.observer = original, observer

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, command):
        phases = {'start': 'native_open', 'continue': 'native_prefill', 'generate_own': 'native_generate'}
        self.observer.phase = phases.get(command.get('op'), 'stream')
        if command.get('op') == 'generate_own':
            self.observer.attempts += 1
        result = self.original(command)
        if command.get('op') == 'generate_own':
            self.observer.generated += len(result['ids'])
            self.observer.phase = 'stream_collect'
        return result


class Tokenizer:
    def __init__(self, original, observer):
        self.original, self.observer = original, observer

    def __getattr__(self, name):
        return getattr(self.original, name)

    def decode(self, *args, **kwargs):
        self.observer.phase = 'decode'
        return self.original.decode(*args, **kwargs)


class ObservedStudio:
    def __init__(self, backend, path):
        self.backend_impl, self.diagnostics = backend, PrivateDiagnostics(path)
        self.phase, self.attempts, self.generated = 'open', 0, 0
        backend.handle = Handle(backend.handle, self)
        backend.tokenizer = Tokenizer(backend.tokenizer, self)
        original = backend.parse_tools
        def parse(*args, **kwargs):
            self.phase = 'tool_parse'
            result = original(*args, **kwargs)
            self.phase = 'stream_serialize'
            return result
        backend.parse_tools = parse

    def _record(self, error):
        backend = self.backend_impl
        counts = {'generation_attempts': self.attempts, 'generated_tokens': self.generated,
                  'input_tokens': backend.input_tokens, 'total_tokens': backend.total_tokens}
        try: self.diagnostics.record(self.phase, error, counts)
        except Exception: pass

    def __getattr__(self, name):
        original = getattr(self.backend_impl, name)
        if name not in {'open', 'own_prompt', 'tool_result'}:
            return original
        def operation(*args, **kwargs):
            self.phase = name
            try: return original(*args, **kwargs)
            except Exception as error:
                self._record(error)
                raise
        return operation

    def stream(self, maximum):
        self.phase = 'stream'
        try: yield from self.backend_impl.stream(maximum)
        except Exception as error:
            self._record(error)
            raise

    def protocol_failure(self, phase, error, counts):
        self.diagnostics.record(phase, error, counts)

    def close(self):
        try: return self.backend_impl.close()
        finally: self.diagnostics.close()


def instrument_studio(backend, path):
    return ObservedStudio(backend, path)
