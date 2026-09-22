"""Accept only versioned snapshot publication on the GLM owner control connection."""
from drift.serving.worker_owner_socket import OwnerSocket
from drift.serving.glm_snapshot_receipts import applied_snapshot


class SnapshotSocket(OwnerSocket):
    def __init__(self, path, deadline, bank, restorer=None):
        self.bank, self.restorer = bank, restorer
        super().__init__(path, deadline)

    def _dispatch(self, frame):
        if type(frame) is dict and set(frame) == {'op', 'version', 'sha256'} and frame['op'] == 'applied':
            if self.restorer is None:
                raise ValueError('SNAPSHOT_RECEIPTS_UNAVAILABLE')
            return applied_snapshot(self.bank, self.restorer, frame['version'], frame['sha256'])
        if type(frame) is not dict or set(frame) != {'op', 'snapshot'} or frame['op'] != 'publish':
            raise ValueError('SNAPSHOT_OWNER_FRAME')
        return {'op': 'published', **self.bank.publish(frame['snapshot'])}

    def _serve(self):
        try: super()._serve()
        finally:
            if self.failed.is_set(): self.bank.close()
