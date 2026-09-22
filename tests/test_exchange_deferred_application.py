"""Separate inbox delivery from cache application while the native receiver is parked."""
import pytest

from drift.exchange.contract import validate_exchange
from drift.exchange.coordinator import dispatch
from drift.exchange.session import ExchangeError
from drift.exchange.lifetime import request_scope
from test_exchange_session import build


def test_delivery_returns_without_waiting_for_a_parked_receiver():
    session = build()
    calls = []

    def parked(*args):
        calls.append(args)
        raise RuntimeError('receiver cannot apply until its next turn')

    session.link.confirm_applied = parked
    reply = dispatch({'r': session}, {'op': 'deliver', 'route': 'r'})
    assert reply['op'] == 'delivered'
    assert reply['delivery']['state'] == 'DELIVERED_NOT_APPLIED'
    assert reply['delivery']['sequence'] == 0
    assert calls == [] and len(session.link.delivered) == 1
    assert session.sequence == session.published_rows == 0
    assert not session.poisoned


def test_only_confirmation_advances_the_cursor_and_can_validate_as_applied():
    session = build()
    delivered = dispatch({'r': session}, {'op': 'deliver', 'route': 'r'})
    pins = dict(session=session.session, sequence=0, mode='drift', source_worker='qwen',
                target_worker='glm', expected_ranks=session.ranks)
    with pytest.raises(ValueError):
        validate_exchange(delivered['delivery'], **pins)
    reply = dispatch({'r': session}, {'op': 'confirm', 'route': 'r', 'sequence': 0})
    assert reply['op'] == 'confirmed'
    record = reply['published']
    record['applied_ranks'], record['receipts'] = tuple(record['applied_ranks']), tuple(record['receipts'])
    assert validate_exchange(record, **pins) == 12
    assert session.sequence == 1 and session.published_rows == 12
    assert len(session.link.delivered) == 1 and session.source.calls == 1


@pytest.mark.parametrize('action', ['deliver', 'exchange'])
def test_pending_delivery_cannot_be_overwritten_or_republished(action):
    session = build()
    dispatch({'r': session}, {'op': 'deliver', 'route': 'r'})
    with pytest.raises(ExchangeError, match='PENDING'):
        dispatch({'r': session}, {'op': action, 'route': 'r'})
    assert session.poisoned and session.source.calls == 1
    assert len(session.link.delivered) == 1


@pytest.mark.parametrize('sequence', [True, -1, 1, '0'])
def test_confirmation_must_bind_the_exact_pending_sequence(sequence):
    session = build()
    calls = []
    session.link.confirm_applied = lambda *args: calls.append(args)
    dispatch({'r': session}, {'op': 'deliver', 'route': 'r'})
    with pytest.raises(ExchangeError):
        dispatch({'r': session}, {'op': 'confirm', 'route': 'r', 'sequence': sequence})
    assert session.poisoned and calls == [] and session.sequence == 0


def test_missing_rank_receipt_never_confirms_a_delivered_publication():
    session = build()
    dispatch({'r': session}, {'op': 'deliver', 'route': 'r'})
    session.link.confirm_applied = lambda *_: dict(ranks=('spark-a.invalid',), receipts=('b' * 64,))
    with pytest.raises(ExchangeError):
        dispatch({'r': session}, {'op': 'confirm', 'route': 'r', 'sequence': 0})
    assert session.poisoned and session.sequence == session.published_rows == 0


def test_confirmation_without_delivery_and_replayed_confirmation_fail_closed():
    for delivered in (False, True):
        session = build()
        if delivered:
            dispatch({'r': session}, {'op': 'deliver', 'route': 'r'})
            dispatch({'r': session}, {'op': 'confirm', 'route': 'r', 'sequence': 0})
        with pytest.raises(ExchangeError):
            dispatch({'r': session}, {'op': 'confirm', 'route': 'r', 'sequence': 0})
        assert session.poisoned


def test_deadline_between_delivery_and_confirmation_prevents_receipt_reads():
    session = build()
    session.deliver_own()
    calls = []
    session.link.confirm_applied = lambda *args: calls.append(args)

    def expired():
        raise ExchangeError('EXCHANGE_DEADLINE')

    with request_scope(expired), pytest.raises(ExchangeError, match='DEADLINE'):
        session.confirm_own(0)
    assert session.poisoned and calls == [] and session.sequence == 0


def test_reply_mutation_does_not_change_the_retained_publication():
    session = build()
    delivery = session.deliver_own()
    delivery.update(rows=1, sequence=12, source_sha256='f' * 64)
    result = session.confirm_own(0)
    assert result['rows'] == 12 and result['sequence'] == 0
    assert result['source_sha256'] != 'f' * 64


def test_deferred_delivery_preserves_capacity_and_text_mode_guards():
    session = build(publish_row_cap=12)
    session.deliver_own()
    session.confirm_own(0)
    with pytest.raises(ExchangeError, match='PUBLISH_CAP'):
        session.deliver_own()
    assert len(session.link.delivered) == 1
    for operation in ('deliver_own', 'confirm_own'):
        session = build(mode='text')
        with pytest.raises(ExchangeError, match='MODE'):
            getattr(session, operation)(*(() if operation == 'deliver_own' else (0,)))
        assert session.link.delivered == [] and session.source.calls == 0
