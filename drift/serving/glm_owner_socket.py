"""Accept only versioned snapshot publication on the GLM owner control connection."""
from drift.serving.worker_owner_socket import OwnerSocket


class SnapshotSocket(OwnerSocket):
    def __init__(self, path, deadline, bank):
        self.bank = bank
        super().__init__(path, deadline)

    def _dispatch(self, frame):
        if type(frame) is not dict or set(frame) != {'op', 'snapshot'} or frame['op'] != 'publish':
            raise ValueError('SNAPSHOT_OWNER_FRAME')
        return {'op': 'published', **self.bank.publish(frame['snapshot'])}

    def _serve(self):
        try: super()._serve()
        finally:
            if self.failed.is_set(): self.bank.close()
