"""Characterize blocked wake semantics with real dispatch and synthetic logits, not model evidence."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drift.serving.live_receiver_studio import append_memory
from drift.serving.worker_activation import ActivationControl
from drift.serving.worker_native_gate import NativeGate
from drift.serving.worker_session import WorkerSession
from drift.serving.worker_studio import continue_native, generate_native
from drift.serving.worker_studio_backend import StudioBackend
from test_worker_session import LIMITS, PINS, command


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]

    def decode(self, ids, **kwargs):
        return ','.join(map(str, ids))


class SyntheticBoundary:
    def __init__(self, tmp_path, *, initial_choice=3, allow_activation_wake=False):
        self.state, self.calls, self.operations = {}, [], []
        self.initial_choice, self.memory_choice = initial_choice, None
        self.tokenizer = Tokenizer()
        namespace = dict(S=self.state, run=self.run, np=np, KV=[0], KV_LAYOUTS={0: (1, 2)},
                         lm=SimpleNamespace(make_cache=lambda: [SimpleNamespace(offset=0, keys=None, values=None)]),
                         rope=None, INDEX_DIM=2, STOP={9}, tok=self.tokenizer,
                         mx=SimpleNamespace(bfloat16=np.float16, eval=lambda *values: None),
                         append_memory=append_memory, append_entries=self.append,
                         continue_native=continue_native, generate_native=generate_native)
        source = Path('scripts/live/studio_drift_worker.py').read_text()
        handle = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'handle')
        exec(compile(ast.Module(body=[handle], type_ignores=[]), '<real-worker-dispatch>', 'exec'), namespace)

        def dispatch(frame):
            self.operations.append(frame['op'])
            return namespace['handle'](frame)

        self.gate = NativeGate(dispatch, self.state, lambda: None)
        self.backend = StudioBackend(self.gate, self.state, self.tokenizer, lambda text, *args: (text, None), reserve=8)
        self.session = WorkerSession('qwen', PINS, LIMITS, self.backend, allow_activation_wake=allow_activation_wake)
        root = tmp_path / 'memory'
        root.mkdir(mode=0o700)
        config = dict(session='s1', source_worker='glm', target_worker='qwen', memory_root=str(root),
                      max_bytes=65536, max_rows=4, max_total_rows=4)
        self.activation = ActivationControl(config, self.gate, {'k0': (1, 2), 'v0': (1, 2)})
        path = root / 'incoming.npz'
        np.savez(path, k0=np.zeros((2, 1, 2), np.float16), v0=np.full((2, 1, 2), 4, np.float16))
        self.publication = dict(v=1, session='s1', seq=1, op='append', path=path.name,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=2)
        reply = self.events(0, 'open', {**PINS, 'limits': LIMITS, 'system_prompt': [], 'tools': []})
        assert reply[0]['op'] == 'opened'

    def run(self, ids):
        state = self.state
        self.calls.append((list(ids), state['foreign_pos']))
        slot = len(state['slot_pos'])
        state['own_slots'].extend(range(slot, slot + len(ids)))
        state['slot_pos'].extend(range(state['own_pos'], state['own_pos'] + len(ids)))
        state['own_pos'] += len(ids)
        state['cache'][0].offset += len(ids)
        logits = np.zeros(10)
        logits[self.initial_choice if self.memory_choice is None else self.memory_choice] = 1
        state['last'] = logits

    def append(self, caches, entries, rope, positions, index_dim, dtype):
        self.memory_choice = int(entries[0][1][0, 0, 0])
        caches[0].offset += len(positions)

    def events(self, sequence, operation, payload):
        return list(self.session.handle(command(sequence, operation, payload)))

    def publish(self):
        return self.activation.apply(self.publication)


def test_applied_receipt_does_not_refresh_first_token_logits(tmp_path):
    fixture = SyntheticBoundary(tmp_path)
    fixture.events(1, 'own_prompt', {'text': 'fixed synthetic local setup'})
    last = fixture.state['last']
    receipt = fixture.publish()
    assert receipt['op'] == 'appended' and receipt['rows'] == 2
    assert fixture.state['last'] is last and fixture.state['last'].argmax() == 3
    events = fixture.events(2, 'stream', {'max_tokens': 2})
    assert events[0]['payload']['text'] == '3,4'
    assert fixture.calls == [([1, 2], 0), ([3], 2), ([4], 2)]
    assert events[-1]['payload']['usage'] == {'input_tokens': 2, 'output_tokens': 2}
    assert fixture.session.tokens == 4


def test_receipt_before_local_query_changes_first_token_without_duplicating_cache(tmp_path):
    fixture = SyntheticBoundary(tmp_path)
    fixture.publish()
    fixture.events(1, 'own_prompt', {'text': 'fixed synthetic local setup'})
    events = fixture.events(2, 'stream', {'max_tokens': 1})
    assert events[0]['payload']['text'] == '4'
    assert fixture.calls == [([1, 2], 2), ([4], 2)]
    assert fixture.state['own_slots'] == [2, 3, 4]
    assert fixture.state['own_pos'] == 11 and fixture.state['cache'][0].offset == 5
    assert events[-1]['payload']['usage'] == {'input_tokens': 2, 'output_tokens': 1}


def test_activation_receipt_alone_cannot_admit_fresh_worker_generation(tmp_path):
    fixture = SyntheticBoundary(tmp_path)
    fixture.publish()
    assert fixture.state['last'] is None
    assert not fixture.session.ready and not fixture.session.control_ready
    events = fixture.events(1, 'stream', {'max_tokens': 1})
    assert events[0]['op'] == 'error' and fixture.session.poisoned
    assert fixture.calls == [] and 'generate_own' not in fixture.operations
    assert fixture.session.turns == 0 and fixture.session.tokens == 0


def test_stopped_worker_keeps_boundary_deferred_and_refuses_unprompted_restream(tmp_path):
    fixture = SyntheticBoundary(tmp_path, initial_choice=9)
    fixture.events(1, 'own_prompt', {'text': 'fixed synthetic local setup'})
    first = fixture.events(2, 'stream', {'max_tokens': 2})
    assert first[-1]['payload']['usage'] == {'input_tokens': 2, 'output_tokens': 1}
    assert fixture.state['done'] and fixture.state['pending_stop'] == 9
    fixture.publish()
    before = (fixture.state['own_pos'], list(fixture.state['own_slots']), len(fixture.calls))
    result = fixture.gate({'op': 'generate_own', 'tokens': 1})
    assert result['ids'] == [] and result['done'] and result['stop_id'] == 9
    assert before == (fixture.state['own_pos'], fixture.state['own_slots'], len(fixture.calls))
    assert fixture.backend.codec.consumed == [1, 2, 9]
    operations = fixture.operations.count('generate_own')
    rejected = fixture.events(3, 'stream', {'max_tokens': 1})
    assert rejected[0]['op'] == 'error' and fixture.session.poisoned
    assert fixture.operations.count('generate_own') == operations
    assert fixture.session.tokens == 3 and fixture.session.turns == 1
