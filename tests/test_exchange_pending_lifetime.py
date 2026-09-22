"""A deferred publication survives clean reconnects but never its coordinator owner."""
import tempfile
import time
from pathlib import Path

import pytest

from drift.exchange.coordinator import ExchangeCoordinator
from drift.exchange.session import ExchangeError
from tests.test_exchange_coordinator import make_session, talk


def test_reconnect_preserves_pending_publication_and_confirm_does_not_republish():
    with tempfile.TemporaryDirectory(prefix='dx-', dir='/tmp') as directory:
        session = make_session()
        coordinator = ExchangeCoordinator(Path(directory) / 'x.sock', {'r': session}, time.monotonic() + 5)
        coordinator.start()
        try:
            delivered = talk(coordinator.path, [{'op': 'deliver', 'route': 'r'}])[0]
            assert delivered['op'] == 'delivered' and session.sequence == 0
            confirmed = talk(coordinator.path, [{'op': 'confirm', 'route': 'r', 'sequence': 0}])[0]
            assert confirmed['op'] == 'confirmed' and session.sequence == 1
            assert session.source.calls == len(session.link.delivered) == 1
        finally:
            coordinator.close()


@pytest.mark.parametrize('reason', ['close', 'deadline'])
def test_owner_exit_invalidates_pending_confirmation(reason):
    with tempfile.TemporaryDirectory(prefix='dx-', dir='/tmp') as directory:
        session = make_session()
        coordinator = ExchangeCoordinator(Path(directory) / 'x.sock', {'r': session}, time.monotonic() + 5)
        coordinator.start()
        try:
            talk(coordinator.path, [{'op': 'deliver', 'route': 'r'}])
            if reason == 'deadline':
                coordinator.deadline = time.monotonic() - 1
                coordinator.thread.join(1)
                assert not coordinator.thread.is_alive()
            else:
                coordinator.close()
            with pytest.raises(ExchangeError, match='POISONED'):
                session.confirm_own(0)
            assert session.sequence == 0
        finally:
            coordinator.close()
