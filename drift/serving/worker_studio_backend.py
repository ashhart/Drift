"""Route private worker turns through the persistent Studio worker and checkpoint codec."""
import threading
import time
from drift.serving.worker_prefill import prefill_chunks
from drift.serving.worker_codec import CheckpointCodec
from drift.serving.worker_assistant import parsed_assistant
from drift.serving.worker_contract import require


class StudioBackend:
    backend = 'live'
    capabilities = {'native_state': 'retained', 'tool_calls': True, 'cancellation': True}

    def __init__(self, handle, state, tokenizer, parse_tools, reserve=4096, *, control_format=None):
        self.handle, self.state, self.tokenizer, self.parse_tools = handle, state, tokenizer, parse_tools
        from drift.serving.worker_studio_control import QWEN_CONTROL
        require(control_format in (None, QWEN_CONTROL), 'CAPABILITY')
        self.control_format = control_format
        self.reserve = reserve
        self.cancelled = threading.Event()
        self.finished = threading.Event(); self.finished.set()
        self.input_tokens = self.total_tokens = 0
        self.tool_results = []
        self.awaiting_tools = 0
        self.pending_control = []
        from drift.serving.worker_activation_wake import FreshActivationBootstrap
        self.activation_bootstrap = FreshActivationBootstrap(self)

    def open(self, payload):
        self.limits = payload['limits']
        self.deadline = time.monotonic() + self.limits['deadline_ms'] / 1000
        self.codec = CheckpointCodec(self.tokenizer, payload['system_prompt'], payload['tools'])
        self.handle({'op': 'start', 'reserve': self.reserve})

    def _append(self, messages):
        ids = self.codec.append_many(messages)
        require(self.total_tokens + self.input_tokens + len(ids) < self.limits['max_session_tokens'], 'LIMIT')
        for count in prefill_chunks(self.handle, self.state, ids, self.cancelled, self.deadline):
            self.input_tokens += count

    def own_control(self, payload):
        from drift.serving.worker_own_control import own_control_message
        from drift.serving.worker_studio_control import reader_control_message
        require(not self.pending_control)
        self.pending_control = [reader_control_message(payload) if self.control_format else own_control_message(payload)]

    def own_control_append(self, payload):
        from drift.serving.worker_own_control import own_control_message
        from drift.serving.worker_studio_control import reader_control_message
        render = reader_control_message if self.control_format else own_control_message
        self.pending_control.append(render(payload, additive=True))

    def own_prompt(self, payload):
        self._append([*self.pending_control, {'role': 'user', 'content': payload['text']}])
        self.pending_control = []

    def activation_wake(self, payload, session, worker):
        return self.activation_bootstrap.arm(payload, session, worker)

    def prepare_activation_wake(self, payload):
        self.activation_bootstrap.pin(payload)

    def tool_result(self, payload):
        require(self.awaiting_tools > 0)
        self.tool_results.append({'role': 'tool', 'tool_call_id': payload['call_id'], 'content': payload['text']})
        self.awaiting_tools -= 1
        if not self.awaiting_tools:
            self._append([*self.tool_results, *self.pending_control])
            self.tool_results = []
            self.pending_control = []

    def stream(self, max_tokens):
        self.finished.clear()
        try:
            require(not self.awaiting_tools and not self.tool_results)
            first = self.activation_bootstrap.first(max_tokens) if self.activation_bootstrap.binding is not None else None
            if self.pending_control:
                self._append(self.pending_control)
                self.pending_control = []
            ids, stop, done = [], None, False
            available = self.limits['max_session_tokens'] - self.total_tokens - self.input_tokens
            require(max_tokens <= available, 'LIMIT')
            for index in range(max_tokens):
                if self.cancelled.is_set():
                    return
                result = first if index == 0 and first is not None else self.handle({'op': 'generate_own', 'tokens': 1})
                ids.extend(result['ids']); stop = result['stop_id']; done = result['done']
                if done:
                    break
            if self.cancelled.is_set():
                return
            text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            cleaned, calls = self.parse_tools(text, self.tokenizer, self.codec.tools)
            message, tool_events = parsed_assistant(cleaned, calls)
            self.awaiting_tools = len(calls or [])
            if calls:
                require(done, 'PROTOCOL')
            self.codec.assistant(message, ids, stop, raw_text=text)
            if cleaned:
                yield {'op': 'text', 'payload': {'text': cleaned}}
            for payload in tool_events:
                yield {'op': 'tool_call', 'payload': payload}
            usage = {'input_tokens': self.input_tokens, 'output_tokens': len(ids) + int(stop is not None)}
            self.total_tokens += sum(usage.values()); self.input_tokens = 0
            yield {'op': 'terminal', 'payload': {'reason': 'tool_use' if calls else 'stop' if done else 'length', 'usage': usage}}
        finally:
            self.finished.set()

    def cancel(self, target_seq):
        self.cancelled.set()

    def close(self):
        self.cancelled.set()
        require(self.finished.wait(timeout=1), 'LIMIT')
        if hasattr(self.handle, 'clear'):
            self.handle.clear()
        else:
            self.state.clear()
            self.state['poisoned'] = True


def serve_studio(configuration, checkpoint, handle, state, source, sink, *, settle=lambda: None, layouts=None):
    from transformers import AutoTokenizer
    from mlx_vlm.tool_parsers import _infer_tool_parser, load_tool_module
    from omlx.api.tool_calling import parse_tool_calls
    from drift.serving.worker_session import WorkerSession
    from drift.serving.worker_stdio import serve
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    parser_name = _infer_tool_parser(tokenizer.chat_template)
    require(parser_name is not None, 'CAPABILITY')
    parser = load_tool_module(parser_name)
    tokenizer.has_tool_calling = True
    tokenizer.tool_call_start, tokenizer.tool_call_end = parser.tool_call_start, parser.tool_call_end
    tokenizer.tool_parser = parser.parse_tool_call
    from drift.serving.worker_native_gate import NativeGate
    from drift.serving.worker_activation_server import ActivationServer
    gate = NativeGate(handle, state, settle)
    from drift.serving.worker_studio_control import checkpoint_control_format
    backend = StudioBackend(gate, state, tokenizer, parse_tool_calls, configuration.get('reserve', 4096),
                            control_format=checkpoint_control_format(checkpoint))
    if configuration.get('diagnostics_path') is not None:
        from drift.serving.worker_studio_diagnostics import instrument_studio
        backend = instrument_studio(backend, configuration['diagnostics_path'])
    session = WorkerSession(configuration['worker'], configuration['pins'], configuration['limits'], backend,
                            allow_own_control=configuration.get('experimental_multi_turn', False),
                            allow_activation_wake=configuration.get('experimental_activation_wake', False))
    require(not session.allow_activation_wake or configuration.get('activation') is not None, 'CAPABILITY')
    activation = None
    try:
        if configuration.get('activation') is not None:
            require(bool(layouts), 'CAPABILITY')
            require(configuration['activation']['target_worker'] == configuration['worker'], 'CAPABILITY')
            session.session = configuration['activation']['session']
            activation = ActivationServer(configuration['activation'], gate, layouts, lambda: backend.cancel(None))
            activation.start()
        return serve(session, source, sink)
    finally:
        if activation is not None:
            activation.close()
