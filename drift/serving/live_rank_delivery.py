"""Require host admission and post-write rank receipts without claiming TP atomicity."""
import json
import re
import shlex
import time

PROBE = '''import hashlib,json,pathlib,sys
rank_delivery_status = json.loads(sys.argv[1])
p = rank_delivery_status
root = pathlib.Path('/dev/shm/glm53-handoff')
out = root / 'tp-live-out' / p['run']
status = {'error': any(out.glob('error.rank*')), 'present': False, 'sha256': None, 'ack': None}
if p['sequence'] is not None:
    stem = f"{p['sequence']:06d}"
    name = '.' + stem + '.tmp.npz' if p['temporary'] else stem + '.npz'
    source = root / 'tp-live-in' / p['run'] / name
    status['present'] = source.is_file()
    if status['present']:
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(65536), b''):
                digest.update(block)
        status['sha256'] = digest.hexdigest()
    ack = out / f"ack.{stem}.rank{p['rank']}.json"
    if ack.is_file():
        receipt = json.loads(ack.read_text()) if ack.stat().st_size <= 512 else {}
        keys = {'rank', 'world_size', 'sequence', 'rows', 'sha256'}
        valid = isinstance(receipt, dict) and set(receipt) == keys
        valid = valid and all(type(receipt[k]) is int and receipt[k] >= 0 for k in keys - {'sha256'})
        digest = receipt.get('sha256') if isinstance(receipt, dict) else None
        valid = valid and isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        status['ack'] = receipt if valid else {'invalid': True}
print(json.dumps(status))
'''


class RankDeliveryError(RuntimeError):
    """At least one configured rank lacks valid delivery evidence."""


class RankDelivery:
    """The host list declares exactly one TP rank per host, in rank order."""

    def __init__(self, hosts, run, ssh, timeout, check_failure, pause=time.sleep):
        if not hosts or len(set(hosts)) != len(hosts):
            raise ValueError('rank hosts must be nonempty and unique')
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', run) or run in {'.', '..'}:
            raise ValueError('invalid rank delivery session name')
        self.hosts, self.run, self.ssh = tuple(hosts), run, ssh
        self.timeout, self.check_failure, self.pause = timeout, check_failure, pause

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RankDeliveryError('rank acknowledgement deadline exceeded')
        return remaining

    def _probe(self, rank, sequence=None, temporary=False, deadline=None):
        self.check_failure()
        request = {'run': self.run, 'rank': rank, 'sequence': sequence, 'temporary': temporary}
        command = f'python3 -c {shlex.quote(PROBE)} {shlex.quote(json.dumps(request))}'
        if deadline is None:
            raw = self.ssh(self.hosts[rank], command)
        else:
            raw = self.ssh(self.hosts[rank], command, timeout=self._remaining(deadline))
            self._remaining(deadline)
        if len(raw) > 1024:
            raise RankDeliveryError('oversized rank status')
        status = json.loads(raw)
        if not isinstance(status, dict) or set(status) != {'error', 'present', 'sha256', 'ack'}:
            raise RankDeliveryError('malformed rank status')
        if type(status['error']) is not bool or status['error']:
            raise RankDeliveryError('rank reported failure')
        return status

    def health(self, *, deadline=None):
        for rank in range(len(self.hosts)):
            self._probe(rank, deadline=deadline)

    def admit(self, sequence, digest, *, deadline=None):
        for rank in range(len(self.hosts)):
            status = self._probe(rank, sequence, temporary=rank == 0, deadline=deadline)
            if status['present'] is not True or status['sha256'] != digest:
                raise RankDeliveryError('rank publication missing or digest differs')

    def wait_applied(self, sequence, rows, digest, *, deadline=None):
        deadline = min(deadline, time.monotonic() + self.timeout) if deadline is not None else time.monotonic() + self.timeout
        while True:
            complete, receipts = True, []
            for rank in range(len(self.hosts)):
                status = self._probe(rank, sequence, deadline=deadline)
                if status['present'] is not True or status['sha256'] != digest:
                    raise RankDeliveryError('rank publication disappeared or changed')
                expected = {'rank': rank, 'world_size': len(self.hosts), 'sequence': sequence,
                            'rows': rows, 'sha256': digest}
                if status['ack'] is None:
                    complete = False
                elif status['ack'] != expected or any(type(status['ack'][key]) is not int for key in ('rank', 'world_size', 'sequence', 'rows')):
                    raise RankDeliveryError('rank acknowledgement differs')
                else:
                    receipts.append(dict(status['ack']))
            remaining = self._remaining(deadline)
            if complete:
                return receipts
            self.pause(min(0.05, remaining))
