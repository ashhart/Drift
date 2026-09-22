"""Stage local owner snapshots and distinguish them from actual cache application."""
import hashlib
import os
import time

from drift.exchange.lifetime import checkpoint
from drift.serving.worker_activation_files import private_root


def require(value):
    if not value:
        raise ValueError('OWNER_LINK_CONTRACT')


def exact(value, expected):
    require(type(value) is type(expected))
    if type(expected) is dict:
        require(value.keys() == expected.keys())
        for key in expected: exact(value[key], expected[key])
    elif type(expected) is list:
        require(len(value) == len(expected))
        for received, wanted in zip(value, expected): exact(received, wanted)
    else:
        require(value == expected)


def write_payload(root, name, body):
    require(type(body) is bytes and 0 < len(body) <= 1048576)
    path = private_root(root) / name
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(body)
    return path, hashlib.sha256(body).hexdigest()


class SnapshotOwnerLink:
    def __init__(self, peer, root, *, session, source_worker, target_worker, recipe, ranks, first_version=2):
        self.peer, self.root, self.session = peer, private_root(root), session
        self.source, self.target, self.recipe = source_worker, target_worker, recipe
        self.ranks, self.first_version, self.staged = tuple(sorted(ranks)), first_version, 0
        require(len(self.ranks) == 2 and len(set(self.ranks)) == 2 and first_version >= 2)

    def stage(self, session, sequence, body, copies):
        require(session == self.session and sequence == self.staged and copies == 1)
        path, digest = write_payload(self.root, f'epoch-{sequence:06d}.npz', body)
        import numpy as np
        with np.load(path, allow_pickle=False) as arrays:
            require(bool(arrays.files))
            rows = arrays[arrays.files[0]].shape[0]
        frame = dict(v=1, session=session, version=self.first_version+sequence, path=path.name,
                     sha256=digest, rows=rows, source_worker=self.source, target_worker=self.target,
                     recipe_sha256=self.recipe)
        reply = self.peer.request({'op':'publish', 'snapshot':frame})
        expected = {key:value for key,value in frame.items() if key not in ('v','path')}
        exact(reply, {'op':'published', **expected, 'bytes':len(body)})
        self.staged += 1
        return dict(version=frame['version'], sha256=digest, rows=rows)

    def confirm_applied(self, staged, rows):
        require(rows == staged['rows'])
        while True:
            checkpoint()
            result = self.peer.request({'op':'applied', 'version':staged['version'], 'sha256':staged['sha256']})
            require(type(result) is dict and set(result) == {'op','state','version','sha256','rows','receipts'}
                    and result['op'] == 'application' and result['version'] == staged['version']
                    and result['sha256'] == staged['sha256'] and result['rows'] == rows)
            require(type(result['version']) is int and type(result['rows']) is int)
            if result['state'] == 'APPLIED':
                exact(result['receipts'], [dict(rank=rank, world_size=2, sequence=0, rows=rows,
                                               sha256=staged['sha256']) for rank in (0,1)])
                return dict(ranks=self.ranks, receipts=(staged['sha256'],)*2)
            require(result['state'] == 'PENDING' and result['receipts'] == [])
            time.sleep(.01)


class AppendOwnerLink:
    def __init__(self, peer, root, *, rank, session, source_worker, target_worker):
        self.peer, self.root, self.rank = peer, private_root(root), rank
        self.session, self.source, self.target = session, source_worker, target_worker
        self.sequence = self.total = 0

    def stage(self, session, sequence, body, copies):
        require(session == self.session and type(sequence) is int and sequence == self.sequence and copies == 1)
        path, digest = write_payload(self.root, f'incoming-{sequence:06d}.npz', body)
        import numpy as np
        with np.load(path, allow_pickle=False) as arrays:
            rows = arrays[arrays.files[0]].shape[0]
        reply = self.peer.request(dict(op='append', path=path.name, sha256=digest, rows=rows))
        exact(reply, dict(v=1, session=self.session, seq=self.peer.requests, source_worker=self.source,
                         target_worker=self.target, op='appended', sha256=digest,
                         rows=rows, foreign_total=self.total+rows))
        self.sequence += 1; self.total += rows
        return dict(rows=rows, sha256=digest)

    def confirm_applied(self, staged, rows):
        require(rows == staged['rows'])
        return dict(ranks=(self.rank,), receipts=(staged['sha256'],))
