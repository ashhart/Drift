"""Bind controller-owned snapshot versions to the existing native GLM turn seam."""
import threading
from drift.serving.glm_restore_turn import RestoringTransport, TurnRestorer
from drift.serving.glm_snapshot_bank import require
from drift.serving.live_publication_transport import PublicationTransport


class BankRestorer(TurnRestorer):
    def __init__(self, previous, bank, route):
        self.bank, self.route, self.captured, self.boundary = bank, route, None, threading.RLock()
        self.publication_factory = getattr(previous, 'publication_factory', None)
        super().__init__(previous.verifier, self.publication, rows=previous.rows, digest=previous.digest,
                         root=previous.root, deadline=previous.deadline, max_input_bytes=previous.max_input_bytes,
                         linked=True, clock=previous.clock, outbox=previous.outbox,
                         causal_prefill=previous.causal_prefill)

    def remaining(self):
        require(not self.bank.failed)
        return super().remaining()

    def publication(self, name):
        captured = self.bank.validate(self.captured)
        if self.publication_factory is not None:
            return self.publication_factory(captured.path, captured.sha256, name, self.remaining)
        return self.route.bind(PublicationTransport(captured.path, captured.sha256, name, self.route.spec['peer'], self.remaining))

    def prepare(self, body):
        with self.boundary:
            require(not self.failed and self.plan is None)
            try:
                self.captured = self.bank.capture()
                self.rows, self.digest = self.captured.rows, self.captured.sha256
                result = super().prepare(body)
                self.plan['snapshot'] = self.captured.receipt()
                return result
            except Exception:
                self.close()
                raise

    def applied(self):
        self.bank.validate(self.captured)
        return super().applied()

    def finish(self):
        with self.boundary:
            receipt = dict(self.plan['snapshot'])
            super().finish()
            self.completed[-1]['snapshot'] = receipt

    def close(self):
        self.bank.close()
        super().close()


def bind_snapshot_bank(session, bank, route, *, translator_sha256):
    require(session.started is None and not session.poisoned and not hasattr(session, 'snapshot_bank'))
    previous = session.restoration
    require(isinstance(session.transport, RestoringTransport) and session.transport.restorer is previous)
    require(previous.linked and not previous.failed and previous.plan is None and not previous.completed)
    captured = bank.capture()
    require(captured.recipe_sha256 == translator_sha256)
    require(captured.sha256 == previous.digest and captured.rows == previous.rows)
    require(session.restoration_ownership == {'source_worker': captured.source_worker, 'target_worker': captured.target_worker})
    if getattr(previous, 'publication_factory', None) is None:
        route.validate()
    else:
        require(route is None)
    restorer = BankRestorer(previous, bank, route)
    def prepare(body):
        restorer.max_input_bytes = session.limits['max_input_bytes']
        return restorer.prepare(body)
    session.transport.restorer = restorer
    session.restoration, session.prepare_turn, session.snapshot_bank = restorer, prepare, bank
    return session
