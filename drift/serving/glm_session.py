"""Reconstruct a private GLM transcript and restore declared memory before each tool turn."""
import copy
import json
import time

from drift.serving.glm_chat_events import ChatEvents
from drift.serving.glm_diagnostics import GlmDiagnostics
from drift.serving.worker_contract import require


class GlmSession:
    backend = 'live'

    def __init__(self, transport, *, model, prepare_turn=None, recovery=None, clock=time.monotonic, diagnostics=None):
        self.transport, self.model, self.prepare_turn, self.recovery = transport, model, prepare_turn, recovery
        self.diagnostics = diagnostics if diagnostics is not None else GlmDiagnostics()
        self.clock, self.started, self.poisoned = clock, None, False
        self.messages, self.pending, self.tools = [], set(), []
        self.pending_control = []
        self.used = 0
        self.capabilities = {'native_state': 'reconstructed', 'tool_calls': True, 'cancellation': recovery is not None}

    def _check(self):
        if self.poisoned or self.started is None or self.clock() - self.started >= self.limits['deadline_ms'] / 1000:
            raise ValueError('GLM session is unavailable')

    def open(self, payload):
        if self.started is not None or self.poisoned:
            raise ValueError('GLM session cannot be reused')
        self.limits, self.tools = dict(payload['limits']), copy.deepcopy(payload['tools'])
        self.messages = [{'role': 'system', 'content': '\n'.join(payload['system_prompt'])}]
        self.started = self.clock()

    def own_prompt(self, payload):
        self._check()
        if self.pending:
            raise ValueError('outstanding tool result')
        self.messages.extend([*self.pending_control, {'role': 'user', 'content': payload['text']}])
        self.pending_control = []

    def own_control(self, payload):
        from drift.serving.worker_own_control import own_control_message
        self._check()
        require(not self.pending_control)
        self.pending_control = [own_control_message(payload)]

    def own_control_append(self, payload):
        from drift.serving.worker_own_control import own_control_message
        self._check()
        self.pending_control.append(own_control_message(payload, additive=True))

    def tool_result(self, payload):
        self._check()
        if payload['call_id'] not in self.pending:
            raise ValueError('unexpected tool result')
        self.messages.append({'role': 'tool', 'tool_call_id': payload['call_id'], 'content': payload['text']})
        self.pending.remove(payload['call_id'])
        if not self.pending:
            self.messages.extend(self.pending_control)
            self.pending_control = []

    def stream(self, maximum):
        self.diagnostics.begin(self.used, getattr(self, 'limits', {}).get('max_session_tokens', 0))
        try:
            self._check()
            if self.pending:
                raise ValueError('outstanding tool result')
            self.messages.extend(self.pending_control)
            self.pending_control = []
            body = {'model': self.model, 'messages': copy.deepcopy(self.messages), 'max_tokens': maximum,
                    'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True},
                    'chat_template_kwargs': {'enable_thinking': False}}
            if self.tools:
                body.update(tools=[{'type': 'function', 'function': tool} for tool in self.tools], tool_choice='auto')
            if self.prepare_turn is not None:
                extra = self.prepare_turn(copy.deepcopy(body))
                if type(extra) is not dict or not set(extra) <= {'kv_transfer_params', 'chat_template', 'cache_salt', 'vllm_xargs'}:
                    raise ValueError('memory preparation attempted to replace own input')
                body.update(extra)
            if len(json.dumps(body).encode()) > self.limits['max_input_bytes']:
                raise ValueError('own input exceeds limit')
            remaining = self.limits['deadline_ms'] / 1000 - (self.clock() - self.started)
            self.diagnostics.mark('glm_count')
            inputs = self.transport.count_tokens(body, remaining)
            if type(inputs) is not int or inputs < 0:
                raise ValueError('invalid native token count')
            self.diagnostics.mark('glm_budget', counted_tokens=inputs)
            require(inputs < self.limits['max_session_tokens'] - self.used, 'LIMIT')
            self.diagnostics.mark('glm_admit', admitted_tokens=inputs)
            body['max_tokens'] = min(maximum, self.limits['max_session_tokens'] - self.used - inputs)
            self._check()
            parsed = ChatEvents({tool['name'] for tool in self.tools}, self.limits['max_input_bytes'] * 8)
            for event in self.diagnostics.events(self.transport.stream(body, self.limits['deadline_ms'] / 1000 - (self.clock() - self.started))):
                self._check()
                yield from parsed.accept(event)
            self.diagnostics.parsed(parsed, body['max_tokens'])
            message, calls, events = parsed.finish()
            usage = parsed.usage
            self.diagnostics.mark('glm_usage')
            if usage['input_tokens'] != inputs or usage['output_tokens'] > body['max_tokens']:
                raise ValueError('native usage differs from admitted token budget')
            self.used += sum(usage.values())
            self.messages.append(message); self.pending = calls
            yield from events
        except Exception as error:
            self.diagnostics.record(error)
            self.close()
            raise

    def protocol_failure(self, phase, error, counts):
        self.diagnostics.protocol_failure(phase, error, counts)

    def cancel(self, target_seq):
        self.transport.cancel()
        self.poisoned = True
        if self.recovery is None:
            raise ValueError('server cancellation is not qualified')
        self.recovery()

    def close(self):
        self.poisoned = True
        try: self.transport.close()
        finally: self.diagnostics.close()
