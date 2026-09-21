"""Check controlled one-row fixture conversion with synthetic taps and translator doubles."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from drift.serving.qwen_fixture import PUBLIC_NOTE, REVERSE_SHA256, build_fixture, sha256_file


def test_unpinned_translator_is_rejected_before_reading_taps(tmp_path):
    weights = tmp_path / 'wrong.npz'
    weights.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='translator hash'):
        build_fixture(tmp_path/'absent', weights, tmp_path/'fixture.npz', 1)


def test_final_source_row_is_translated_once_then_copied_twelve_times(tmp_path, monkeypatch):
    import drift.serving.qwen_fixture as module
    tap = tmp_path / 'tap.npz'
    arrays = {f'{kind}{3+4*i}': np.stack([np.zeros((2,256)),np.ones((2,256))]).astype(np.float16)
              for i in range(12) for kind in ('k','v')}
    np.savez(tap, **arrays)
    weights = tmp_path/'weights'
    weights.write_bytes(b'synthetic-reader')
    original = sha256_file
    monkeypatch.setattr(module,'sha256_file',lambda path: REVERSE_SHA256 if Path(path)==weights else original(path))
    class Reader:
        sha256 = REVERSE_SHA256
        def read(self, values, gain):
            assert gain == 1.0 and len(values) == 12
            assert all(value.shape == (1,1024) and np.all(value == 1) for value in values.values())
            return {3+4*i: np.full((1,512),i,dtype=np.float32) for i in range(11)}
    monkeypatch.setattr(module.StackedReader,'load',lambda *a,**kw: Reader())
    output = tmp_path/'fixture.npz'
    receipt = build_fixture(tap,weights,output,2)
    with np.load(output) as result:
        assert len(result.files)==11
        assert all(result[name].shape==(12,512) for name in result.files)
        assert np.all(result['l7']==1)
    assert receipt['source_row']==1 and receipt['source_tokens']==2 and receipt['generated_tokens']==0
    assert receipt['fixture_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt['source_text_sha256']==hashlib.sha256(PUBLIC_NOTE.encode()).hexdigest()
    assert 'API:' not in str(receipt)


@pytest.mark.parametrize('tokens', [0,33,True])
def test_source_token_limit_fails_before_reading_files(tmp_path,tokens):
    with pytest.raises(ValueError,match='token count'):
        build_fixture(tmp_path/'tap',tmp_path/'weights',tmp_path/'output',tokens)


def test_aggregate_receipt_rejects_extra_text_fields():
    from drift.serving.fixture_receipt import validate_receipt
    with pytest.raises(ValueError,match='fields'):
        validate_receipt({'raw_text':'PRIVATE_SYNTHETIC'})
