"""Fail closed on any exchange record that does not evidence a bounded verified transfer."""
import pytest

from drift.exchange.contract import validate_exchange


def record(**overrides):
    value = dict(v=1, session='live-1', sequence=0, mode='drift', source_worker='qwen', target_worker='glm',
                 rows=12, source_rows=1, copies=12, bytes=4096, source_sha256='a' * 64,
                 text_bytes=0, applied_ranks=('spark-a.invalid', 'spark-b.invalid'), receipts=('b' * 64, 'c' * 64))
    value.update(overrides)
    return value


def bound():
    return dict(session='live-1', sequence=0, mode='drift', source_worker='qwen', target_worker='glm',
                expected_ranks=('spark-a.invalid', 'spark-b.invalid'))


def test_a_complete_drift_record_validates():
    assert validate_exchange(record(), **bound()) == 12


@pytest.mark.parametrize('field,value', [
    ('session', 'other'), ('sequence', 1), ('mode', 'text'), ('source_worker', 'glm'),
    ('target_worker', 'qwen'), ('rows', 0), ('source_rows', 0), ('bytes', 0),
    ('source_sha256', 'zz'), ('copies', 0),
])
def test_every_bound_field_is_checked(field, value):
    with pytest.raises(ValueError):
        validate_exchange(record(**{field: value}), **bound())


def test_drift_mode_refuses_any_text_crossing():
    with pytest.raises(ValueError):
        validate_exchange(record(text_bytes=1), **bound())


def test_a_rank_that_did_not_apply_is_not_a_transfer():
    with pytest.raises(ValueError):
        validate_exchange(record(applied_ranks=('spark-a.invalid',), receipts=('b' * 64,)), **bound())
    with pytest.raises(ValueError):
        validate_exchange(record(applied_ranks=('spark-a.invalid', 'zgx3')), **bound())


def test_a_receipt_per_applied_rank_is_required():
    with pytest.raises(ValueError):
        validate_exchange(record(receipts=('b' * 64,)), **bound())
    with pytest.raises(ValueError):
        validate_exchange(record(receipts=('b' * 64, 'not-a-digest')), **bound())


def test_rows_must_cover_every_copy_of_the_source():
    with pytest.raises(ValueError):
        validate_exchange(record(rows=11), **bound())


def test_text_mode_carries_no_rows_and_no_receipts():
    text = dict(bound(), mode='text')
    assert validate_exchange(record(mode='text', rows=0, copies=0, source_rows=0, source_sha256=None,
                                    bytes=0, text_bytes=128, applied_ranks=(), receipts=()), **text) == 0
    with pytest.raises(ValueError):
        validate_exchange(record(mode='text', text_bytes=128), **text)


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValueError):
        validate_exchange(dict(record(), extra=1), **bound())
