import time
from types import SimpleNamespace

import numpy as np
import pytest

from drift.serving.bridge_files import QWEN_LAYERS
from drift.transcript.qwen import QwenReader


@pytest.fixture
def reader(monkeypatch):
    value = object.__new__(QwenReader)
    arrays = {layer: (np.zeros((2, 2, 256), np.float32), np.ones((2, 2, 256), np.float32)) for layer in QWEN_LAYERS}
    keys = {layer: np.ones((2, 4), np.float32) for layer in QWEN_LAYERS}
    value.reader = SimpleNamespace(read=lambda x: (arrays, keys, np.array([0, 1]), None))
    value.mx = SimpleNamespace(array=np.array, arange=np.arange, int32=np.int32,
                               bfloat16=np.float32, eval=lambda x: None, argmax=np.argmax)
    value.tokenizer = SimpleNamespace(encode=lambda text, **k: SimpleNamespace(ids=[1, 2, 3]),
                                      decode=lambda ids: 'decoded')
    events = []
    class Model:
        def make_cache(self):
            events.append('cache')
            return {layer: SimpleNamespace(keys=None) for layer in QWEN_LAYERS}
        def __call__(self, ids, **kwargs):
            events.append(('step', kwargs['position_ids'].tolist()))
            logits = np.zeros((1, ids.shape[1], 8))
            logits[..., 4] = 1
            return SimpleNamespace(logits=logits)
    value.lm = Model()
    value.index_dim, value.context_limit, value.rope = 4, 50, None
    value.stop, value.pins = {7}, {}
    monkeypatch.setattr('drift.transcript.qwen.append_entries', lambda *a, **k: events.append('append'))
    return value, events, keys


def test_qwen_injects_before_question_and_obeys_decode_budget(reader):
    value, events, _ = reader
    result = value.answer({3: np.zeros((2, 512))}, 'question', max_new=2,
                          max_rows=10, deadline=time.monotonic() + 5)
    assert events == ['cache', 'append', ('step', [[2, 3, 4]]), ('step', [[5]])]
    assert result['generated_tokens'] == 2 and result['finish_reason'] == 'length'
    assert result['cache_applied'] and result['memory_rows'] == 2


@pytest.mark.parametrize('failure', ['capacity', 'translation', 'deadline'])
def test_qwen_rejects_before_cache_mutation(reader, failure):
    value, events, keys = reader
    maximum, deadline = 10, time.monotonic() + 5
    if failure == 'capacity':
        maximum = 1
    elif failure == 'translation':
        keys.pop(QWEN_LAYERS[-1])
    else:
        deadline = time.monotonic() - 1
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        value.answer({3: np.zeros((2, 512))}, 'question', max_new=2, max_rows=maximum, deadline=deadline)
    assert events == []
