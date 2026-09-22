import numpy as np
import pytest

from drift.serving.bridge_files import GLM_LAYERS
from drift.transcript.blob import encode, decode


def latents(rows=3):
    return {layer: np.full((rows, 512), layer / 100, np.float32) for layer in GLM_LAYERS}


def test_tensor_only_roundtrip():
    values = latents()
    blob = encode(values)
    assert b'prompt_token_ids' not in blob and b'content' not in blob
    restored = decode(blob)
    for layer in values:
        np.testing.assert_array_equal(restored[layer], values[layer])


@pytest.mark.parametrize('corrupt', [lambda b: b[:-1], lambda b: b + b'text', lambda b: b'?' + b[1:]])
def test_blob_requires_exact_shape_and_no_trailing_payload(corrupt):
    with pytest.raises(ValueError):
        decode(corrupt(encode(latents())))


def test_missing_layer_nan_and_oversize_rejected():
    values = latents()
    values.pop(GLM_LAYERS[0])
    with pytest.raises(ValueError):
        encode(values)
    values = latents()
    values[GLM_LAYERS[0]][0, 0] = np.nan
    with pytest.raises(ValueError):
        encode(values)
    with pytest.raises(ValueError):
        decode(encode(latents()), max_rows=2)
