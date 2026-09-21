import time
import pytest
from drift.serving.worker_contract import WorkerError
from drift.serving.worker_native_gate import NativeGate
from drift.serving.worker_studio_backend import StudioBackend
from drift.serving.worker_studio import continue_native


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(6667))


def backend_for(run, limit=20000):
    state = {'cache': object()}
    def handle(frame):
        if frame['op'] == 'start':
            return {}
        return continue_native(state, frame['ids'], run)
    gate = NativeGate(handle, state, lambda: None)
    backend = StudioBackend(gate, state, Tokenizer(), None, reserve=0)
    backend.open({'limits': {'max_session_tokens': limit, 'deadline_ms': 50000}, 'system_prompt': [], 'tools': []})
    return backend, state, gate


def test_actual_duo_sized_prefill_splits_without_changing_tokens():
    chunks = []
    backend, _, _ = backend_for(lambda ids: chunks.append(list(ids)))
    backend.own_prompt({'text': 'Public synthetic child task'})
    assert [len(ids) for ids in chunks] == [4096, 2571]
    assert sum(chunks, []) == list(range(6667))
    assert backend.input_tokens == 6667


def test_entire_prefill_budget_checked_before_first_native_chunk():
    chunks = []
    backend, _, _ = backend_for(lambda ids: chunks.append(ids), limit=6667)
    with pytest.raises(WorkerError, match='LIMIT'):
        backend.own_prompt({'text': 'Public synthetic child task'})
    assert chunks == []
    assert backend.input_tokens == 0


def test_midchunk_failure_poisons_native_state_and_preserves_usage():
    chunks = []
    def run(ids):
        chunks.append(ids)
        if len(chunks) == 2:
            raise RuntimeError('synthetic failure')
    backend, state, gate = backend_for(run)
    with pytest.raises(RuntimeError):
        backend.own_prompt({'text': 'Public synthetic child task'})
    assert [len(ids) for ids in chunks] == [4096, 2571]
    assert state['poisoned'] and gate.failed.is_set()
    assert backend.input_tokens == 4096
    with pytest.raises(WorkerError):
        backend.own_prompt({'text': 'Public synthetic child task'})
    assert len(chunks) == 2


def test_cancellation_between_chunks_stops_and_poisons():
    chunks = []
    def run(ids):
        chunks.append(ids)
        backend.cancel(None)
    backend, state, gate = backend_for(run)
    with pytest.raises(WorkerError):
        backend.own_prompt({'text': 'Public synthetic child task'})
    assert [len(ids) for ids in chunks] == [4096]
    assert backend.input_tokens == 4096
    assert state['poisoned'] and gate.failed.is_set()


def test_deadline_between_chunks_stops_and_poisons(monkeypatch):
    chunks = []
    now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    def run(ids):
        chunks.append(ids)
        now[0] = 51
    backend, state, gate = backend_for(run)
    with pytest.raises(WorkerError, match='LIMIT'):
        backend.own_prompt({'text': 'Public synthetic child task'})
    assert [len(ids) for ids in chunks] == [4096]
    assert backend.input_tokens == 4096
    assert state['poisoned'] and gate.failed.is_set()


def test_pending_stop_consumed_once_across_chunks():
    chunks = []
    backend, state, _ = backend_for(lambda ids: chunks.append(list(ids)))
    state['pending_stop'] = 99
    backend.own_prompt({'text': 'Public synthetic child task'})
    assert chunks[0] == [99, *range(4096)]
    assert chunks[1] == list(range(4096, 6667))
    assert state['pending_stop'] is None
    assert backend.input_tokens == 6667


def test_entire_append_holds_native_transaction(monkeypatch):
    depths = []; depth = [0]
    backend, _, gate = backend_for(lambda ids: depths.append(depth[0]))
    from contextlib import contextmanager
    original = gate.transaction
    @contextmanager
    def tracked():
        with original():
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1
    monkeypatch.setattr(gate, 'transaction', tracked)
    backend.own_prompt({'text': 'Public synthetic child task'})
    assert depths == [2, 2]
