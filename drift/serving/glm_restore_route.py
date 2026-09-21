"""Bind one owner-declared SSH route without changing global SSH configuration."""
import json
from pathlib import Path
import re
from drift.serving.worker_artifacts import digest


def canonical_file(value, maximum):
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('restoration artifact path or size mismatch')
    return path


class PinnedRoute:
    def __init__(self, path, expected):
        self.path, self.expected = canonical_file(path, 65536), expected
        if digest(self.path, 65536) != expected: raise ValueError('route digest mismatch')
        self.spec = json.loads(self.path.read_bytes())
        keys = {'peer', 'hostname', 'identity_file', 'known_hosts_file', 'known_hosts_sha256'}
        if set(self.spec) != keys: raise ValueError('route schema mismatch')
        for key in ('peer', 'hostname'):
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', self.spec[key]):
                raise ValueError('route host mismatch')
        self.validate()

    def validate(self):
        if digest(canonical_file(str(self.path), 65536), 65536) != self.expected:
            raise ValueError('route changed')
        identity = canonical_file(self.spec['identity_file'], 65536)
        if identity.stat().st_mode & 0o077: raise ValueError('route identity permissions too broad')
        hosts = canonical_file(self.spec['known_hosts_file'], 1048576)
        if digest(hosts, 1048576) != self.spec['known_hosts_sha256']: raise ValueError('route host keys changed')

    def bind(self, transport):
        self.validate()
        transport.ssh_options = ['-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                 '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
                                 '-o', 'UserKnownHostsFile=' + self.spec['known_hosts_file'], '-i', self.spec['identity_file']]
        original = transport.stage
        def stage(*, timeout):
            import time
            deadline = time.monotonic() + timeout
            self.validate()
            if transport.probe(self.spec['peer'], 'hostname', timeout=timeout).strip() != self.spec['hostname']:
                raise ValueError('rank route hostname mismatch')
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError('rank route deadline exceeded')
            original(timeout=remaining)
        transport.stage = stage
        return transport
