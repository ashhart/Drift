"""Arm one fresh child using applied KV and its receiver-local checkpoint template."""
from drift.serving.worker_contract import fields, integer, require
from drift.serving.worker_codec import CheckpointCodec
from drift.serving.worker_prefill import prefill_chunks


class FreshActivationBootstrap:
    def __init__(self, backend):
        self.backend, self.binding = backend, None
        self.template_ids = None

    def pin(self, payload):
        require(self.template_ids is None, 'CAPABILITY')
        codec = CheckpointCodec(self.backend.tokenizer, payload['system_prompt'], payload['tools'])
        ids = codec.append_many([])
        require(len(ids) < self.backend.limits['max_session_tokens'], 'LIMIT')
        self.template_ids = tuple(ids)

    def _fresh(self):
        backend, state = self.backend, self.backend.state
        require(self.template_ids is not None, 'CAPABILITY')
        require(not backend.cancelled.is_set() and not state.get('poisoned'), 'WORKER')
        require('cache' in state and not state.get('own_slots') and state.get('last') is None, 'CAPABILITY')
        require(not state.get('done') and state.get('pending_stop') is None, 'CAPABILITY')
        require(not backend.codec.consumed and backend.total_tokens == backend.input_tokens == 0, 'CAPABILITY')
        require(not backend.pending_control and not backend.awaiting_tools and not backend.tool_results, 'CAPABILITY')

    def arm(self, payload, session, worker):
        fields(payload, ('activation_seq', 'foreign_total'))
        sequence = integer(payload['activation_seq'], 1, 2**53 - 1)
        total = integer(payload['foreign_total'], 1, 1_000_000)
        require(self.binding is None and hasattr(self.backend.handle, 'transaction'), 'CAPABILITY')
        binding = {'session': session, 'target_worker': worker, 'activation_seq': sequence, 'foreign_total': total}
        with self.backend.handle.transaction():
            self._fresh()
            require(self.backend.state.get('activation_receipt') == binding, 'CAPABILITY')
            require(self.backend.state.get('foreign_pos') == total, 'CAPABILITY')
            self.binding = binding
        return dict(payload)

    def first(self, max_tokens):
        backend = self.backend
        with backend.handle.transaction():
            self._fresh()
            require(self.binding is not None and backend.state.get('activation_receipt') == self.binding, 'CAPABILITY')
            require(backend.state.get('foreign_pos') == self.binding['foreign_total'], 'CAPABILITY')
            ids = backend.codec.append_many([])
            require(tuple(ids) == self.template_ids, 'CAPABILITY')
            require(len(ids) + max_tokens <= backend.limits['max_session_tokens'], 'LIMIT')
            for count in prefill_chunks(backend.handle, backend.state, ids, backend.cancelled, backend.deadline):
                backend.input_tokens += count
            result = backend.handle({'op': 'generate_own', 'tokens': 1})
            self.binding = None
            return result


def admit_activation_wake(session, payload):
    require(session.allow_activation_wake, 'CAPABILITY')
    require(not session.ready and not session.control_ready and not session.pending, 'CAPABILITY')
    require(session.turns == session.tokens == session.control_updates == 0, 'CAPABILITY')
    require(callable(getattr(session.backend, 'activation_wake', None)), 'CAPABILITY')
    receipt = session.backend.activation_wake(payload, session.session, session.worker)
    session._deadline()
    session.ready = True
    return 'activation_wake_ack', receipt
