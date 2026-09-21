import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import pytest
from drift.serving.worker_cache_settle import settle_native_caches
from drift.serving.worker_native_gate import NativeGate


class Array:
    def __init__(self):
        self.producer = threading.get_ident()
        self.ready = False


class Cache:
    def __init__(self, state):
        self.state = state


def test_real_studio_settle_callback_includes_recurrent_conv_and_ple_states():
    arrays = [Array() for _ in range(9)]
    qsa = Cache(tuple(arrays[:4]))
    qsa.keys, qsa.values, qsa.index_keys, qsa.index_position_ids = arrays[:4]
    qsa._pooled_index_keys = arrays[4]
    recurrent = Cache([arrays[5], arrays[6], [arrays[7], {'ple': arrays[8]}]])
    state = {'cache': [qsa, recurrent]}
    calls = []
    def evaluate(*values): calls.extend(values)
    source = Path('scripts/live/studio_drift_worker.py').read_text()
    function = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == 'settle_native')
    namespace = {'S': state, 'KV': [0], 'mx': SimpleNamespace(array=Array, eval=evaluate), 'settle_native_caches': settle_native_caches}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'settle_callback', 'exec'), namespace)
    namespace['settle_native']()
    assert set(calls) == set(arrays)


def test_producer_materializes_cache_under_native_gate_before_next_thread():
    state = {'cache': [Cache([None] * 4)]}
    def evaluate(*arrays):
        assert gate.lock._is_owned()
        for array in arrays:
            if not array.ready:
                assert array.producer == threading.get_ident()
            array.ready = True
    def handle(command):
        if command['op'] == 'prefill': state['cache'][0].state[:] = [Array() for _ in range(4)]
        else:
            assert all(value.ready for value in state['cache'][0].state)
            state['cache'][0].state[0] = Array()
    gate = NativeGate(handle, state, lambda: settle_native_caches(state['cache'], evaluate, Array))
    gate({'op': 'prefill'})
    errors = []
    def other():
        try:
            gate({'op': 'decode'})
            assert all(value.ready for value in state['cache'][0].state)
        except Exception as error: errors.append(error)
    thread = threading.Thread(target=other); thread.start(); thread.join(1)
    assert not thread.is_alive() and not errors


def test_settlement_avoids_model_references_and_handles_aliases_cycles():
    class Model(dict): pass
    live, parameter = Array(), Array()
    state = {'one': live}; state['self'] = state
    cache = Cache([state, live, None]); cache._pooled_index_tag = Model(weight=parameter)
    seen = []
    assert settle_native_caches([cache], lambda *values: seen.extend(values), Array) == 1
    assert seen == [live]


def test_failed_settlement_poisons_gate():
    state = {'cache': [Cache([Array()])]}
    def fail(*values): raise RuntimeError('synthetic evaluation failure')
    gate = NativeGate(lambda command: None, state, lambda: settle_native_caches(state['cache'], fail, Array))
    with pytest.raises(RuntimeError): gate({'op': 'prefill'})
    assert state['poisoned'] and gate.failed.is_set()
