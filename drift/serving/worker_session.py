"""Bind private own-input operations to one pinned worker and explicit backend semantics."""
import threading
import time
from drift.serving.worker_contract import WorkerError, fields, integer, require, size, validate_event, validate_open


class WorkerSession:
    def __init__(self, worker, pins, ceilings, backend, clock=time.monotonic, *, allow_own_control=False, allow_activation_wake=False):
        require(type(allow_own_control) is bool, 'CAPABILITY')
        require(type(allow_activation_wake) is bool, 'CAPABILITY')
        self.allow_activation_wake = allow_activation_wake
        self.allow_own_control, self.control_updates = allow_own_control, 0
        self.worker, self.pins, self.ceilings, self.backend = worker, dict(pins), dict(ceilings), backend
        self.clock, self.session, self.seq, self.started = clock, None, 0, None
        self.limits = dict(ceilings)
        self.tokens = self.turns = self.output_bytes = 0
        self.poisoned = self.closed = False
        self.ready = self.control_ready = False
        self.tools, self.pending = set(), set()
        self.active_seq = None
        self.lock = threading.Lock()
        self.cancel_requested = threading.Event()
        self.stream_finished = threading.Event()
        self.stream_started = threading.Event()

    def _reply(self, frame, op, payload):
        return {**{key: frame.get(key) for key in ('v', 'session', 'worker', 'seq')}, 'op': op, 'payload': payload}

    def _deadline(self):
        if self.started is not None:
            require((self.clock() - self.started) * 1000 < self.limits['deadline_ms'], 'LIMIT')

    def _admit(self, frame):
        fields(frame, ('v', 'session', 'worker', 'seq', 'op', 'payload'))
        require(type(frame['v']) is int and frame['v'] == 1 and frame['worker'] == self.worker)
        require(type(frame['session']) is str and 0 < len(frame['session']) <= 96)
        require(self.session is None or frame['session'] == self.session)
        require(type(frame['seq']) is int and frame['seq'] == self.seq + 1)
        require(not self.closed and (not self.poisoned or frame['op'] == 'close'))
        require(self.active_seq is None or frame['op'] == 'cancel')
        self._deadline()
        require(size(frame['payload']) <= self.limits['max_input_bytes'], 'LIMIT')
        self.seq = frame['seq']

    def handle(self, frame):
        phase = "protocol_admit"
        try:
            with self.lock:
                self._admit(frame)
                phase = "protocol_operation"
                op, payload = frame['op'], frame['payload']
                if op == 'stream':
                    fields(payload, ('max_tokens',))
                    maximum = integer(payload['max_tokens'], 1, min(self.limits['max_output_tokens'], self.limits['max_session_tokens'] - self.tokens))
                    require((self.ready or self.control_ready) and not self.pending)
                    require(self.turns < self.limits['max_turns'], 'LIMIT')
                    self.active_seq = frame['seq']; self.turns += 1; self.ready = self.control_ready = False
                    self.stream_finished.clear()
                    self.stream_started.set()
                else:
                    response = self._operation(op, payload, frame)
            if op != 'stream':
                yield self._reply(frame, *response)
                return
            phase = "protocol_stream"
            terminal = False
            terminal_event = None
            count = 0
            for event in self.backend.stream(maximum):
                if self.cancel_requested.is_set():
                    return
                self._deadline()
                require(not terminal)
                validate_event(event, self.tools)
                count += 1
                self.output_bytes += size(event)
                require(count <= 4 * maximum + 16 and self.output_bytes <= self.limits['max_input_bytes'] * self.limits['max_turns'], 'LIMIT')
                if event['op'] == 'tool_call':
                    require(self.backend.capabilities['tool_calls'], 'CAPABILITY')
                    call = event['payload']['call_id']; require(call not in self.pending); self.pending.add(call)
                if event['op'] == 'terminal':
                    usage = event['payload']['usage']
                    used = integer(usage['input_tokens'], 0, self.limits['max_session_tokens']) + integer(usage['output_tokens'], 0, maximum)
                    require(self.tokens + used <= self.limits['max_session_tokens'], 'LIMIT')
                    require((event['payload']['reason'] == 'tool_use') == bool(self.pending))
                    self.tokens += used; terminal = True; terminal_event = event
                    continue
                yield self._reply(frame, event['op'], event['payload'])
            require(terminal or self.cancel_requested.is_set())
            self.active_seq = None
            if terminal_event is not None:
                yield self._reply(frame, 'terminal', terminal_event['payload'])
        except Exception as error:
            if type(frame) is dict and frame.get('op') == 'stream' and self.cancel_requested.is_set():
                return
            try:
                from drift.serving.worker_diagnostics import protocol_failure
                protocol_failure(self, frame, phase, error)
            except Exception: pass
            self.poisoned = True
            if self.started is not None:
                try: self.backend.close()
                except Exception: pass
            code = str(error) if isinstance(error, WorkerError) and str(error) in ('PROTOCOL', 'LIMIT', 'CAPABILITY', 'WORKER') else 'WORKER'
            yield self._reply(frame if type(frame) is dict else {}, 'error', {'code': code})
        finally:
            if type(frame) is dict and frame.get('op') == 'stream':
                self.stream_started.set()
                self.stream_finished.set()

    def _operation(self, op, payload, frame):
        if op == 'open':
            require(self.started is None)
            validate_open(payload, self.pins, self.ceilings)
            require(self.backend.backend in ('fixture', 'live'), 'CAPABILITY')
            require(self.backend.capabilities['native_state'] in ('fixture', 'retained', 'reconstructed'), 'CAPABILITY')
            self.limits = dict(payload['limits']); self.session = frame['session']; self.started = self.clock()
            self.tools = {tool['name'] for tool in payload['tools']}
            require(not self.tools or self.backend.capabilities['tool_calls'], 'CAPABILITY')
            self.backend.open(payload); self._deadline()
            if self.allow_activation_wake:
                prepare = getattr(self.backend, 'prepare_activation_wake', None)
                require(callable(prepare), 'CAPABILITY')
                prepare(payload); self._deadline()
            return 'opened', {**self.pins, 'limits': self.limits, 'backend': self.backend.backend, 'capabilities': dict(self.backend.capabilities)}
        require(self.started is not None)
        if op == 'activation_wake':
            from drift.serving.worker_activation_wake import admit_activation_wake
            return admit_activation_wake(self, payload)
        if op in ('own_control', 'own_control_append'):
            from drift.serving.worker_own_control import apply_own_control
            return apply_own_control(self, payload, additive=op == 'own_control_append')
        if op == 'own_prompt':
            fields(payload, ('text',)); require(type(payload['text']) is str and bool(payload['text']))
            require(not self.pending and not self.ready)
            self.backend.own_prompt(payload); self._deadline(); self.ready = True
            return 'own_prompt_ack', {}
        if op == 'tool_result':
            fields(payload, ('call_id', 'text', 'is_error'))
            require(payload['call_id'] in self.pending and type(payload['text']) is str and type(payload['is_error']) is bool)
            self.backend.tool_result(payload); self._deadline(); self.pending.remove(payload['call_id']); self.ready = True
            return 'tool_result_ack', {'call_id': payload['call_id']}
        if op == 'cancel':
            fields(payload, ('target_seq',)); require(payload['target_seq'] == self.active_seq and self.active_seq is not None)
            require(self.backend.capabilities['cancellation'], 'CAPABILITY')
            self.cancel_requested.set()
            self.backend.cancel(payload['target_seq'])
            remaining = self.limits['deadline_ms'] / 1000 - (self.clock() - self.started)
            require(self.stream_finished.wait(max(0, remaining)), 'LIMIT')
            self._deadline(); self.poisoned = True; self.active_seq = None
            return 'cancelled', {'target_seq': payload['target_seq']}
        if op == 'close':
            fields(payload, ()); self.backend.close(); self.closed = True
            return 'closed', {}
        raise WorkerError('PROTOCOL')
