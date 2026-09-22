"""Own pinned dedicated-target connections for one bounded native runtime."""
import ipaddress
import math
import threading
import time

from drift.serving.mcdma_mailbox import Reader, TARGET, TARGET_MAGIC, Window, _get
from drift.serving.mcdma_reverse import ReversePublisher


def require(condition):
    if not condition:
        raise ValueError('MCDMA_TARGET_CONTRACT')


def targets(config):
    require(type(config) is dict and set(config) == {'v', 'targets'} and type(config['v']) is int and config['v'] == 1)
    values = config['targets']
    require(type(values) is list and len(values) == 2)
    fields = {'rank', 'peer', 'source', 'region_len', 'region_vaddr', 'rkey', 'session'}
    for rank, value in enumerate(values):
        require(type(value) is dict and set(value) == fields)
        require(all(type(value[key]) is int for key in fields - {'peer', 'source'}))
        require(value['rank'] == rank and 0 < value['rkey'] < 2**32 and 0 < value['session'] < 2**64)
        require(0 < value['region_vaddr'] < 2**64 and 32768 <= value['region_len'] <= 2**30)
        require(value['region_len'] % 8192 == 0 and value['region_vaddr'] + value['region_len'] <= 2**64)
        for key in ('peer', 'source'):
            require(type(value[key]) is str)
            address = ipaddress.ip_address(value[key])
            require(address.version == 4 and not address.is_unspecified and not address.is_multicast)
    require(values[0]['region_len'] == values[1]['region_len'])
    require(values[0]['peer'] != values[1]['peer'] and values[0]['source'] != values[1]['source'])
    return [dict(value) for value in values]


class OwnedRegion:
    def __init__(self, conn, owner):
        self.conn, self.owner, self.lock = conn, owner, threading.RLock()
        self.region_len = conn.region_len

    def _call(self, method, *args, **kwargs):
        with self.lock:
            self.owner.check()
            result = method(*args, **kwargs)
            self.owner.check()
            return result

    def get(self, length, offset=0):
        return self._call(self.conn.get, length, offset=offset)

    def put(self, data, offset=0):
        return self._call(self.conn.put, data, offset=offset)

    def close(self):
        with self.lock:
            self.conn.close()


class McdmaConnections:
    def __init__(self, config, opener, *, deadline):
        self.closed, self.connections = False, []
        require(type(deadline) in (int, float) and math.isfinite(deadline))
        self.deadline = deadline
        self.check()
        require(deadline - time.monotonic() <= 600)
        specs = targets(config)
        half = specs[0]['region_len'] // 2
        try:
            regions = {spec['rank']: self._open(spec, opener) for spec in specs}
            forward = self._open(specs[0], opener)
            self.publisher = ReversePublisher(regions, head=0, timeout_s=8, split=half)
            self.check()
            self.reader = Reader(Window(forward, half, half))
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.closed or time.monotonic() >= self.deadline:
            raise TimeoutError('MCDMA_RUNTIME_CLOSED')

    def _open(self, spec, opener):
        self.check()
        conn = opener(spec['peer'], src=spec['source'])
        try:
            require(all(getattr(conn, field) == spec[field] for field in ('region_len', 'region_vaddr', 'rkey')))
            region = OwnedRegion(conn, self)
            record = TARGET.unpack(_get(region, 0, TARGET.size))
            require(record == (TARGET_MAGIC, spec['session'], spec['region_len'] // 2))
        except BaseException:
            conn.close()
            raise
        self.connections.append(region)
        return region

    def close(self):
        self.closed = True
        errors = []
        for region in reversed(self.connections):
            try:
                region.close()
            except Exception as error:
                errors.append(error)
        self.connections.clear()
        if errors:
            raise RuntimeError('MCDMA_CONNECTION_CLOSE_FAILED') from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
