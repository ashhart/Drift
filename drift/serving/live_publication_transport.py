"""Stage one private publication on two ranks without exporting its contents."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from drift.serving.live_rank_delivery import RankDelivery

PEER_STAGE = '''import hashlib,json,pathlib,sys
p=json.loads(sys.argv[1]); root=pathlib.Path('/dev/shm/glm53-handoff')
folders=[root/k/p['run'] for k in ('tp-live-in','tp-live-out')]
if any(x.exists() or x.is_symlink() for x in folders): raise SystemExit(2)
blob=sys.stdin.buffer.read(1048577)
if len(blob)>1048576 or hashlib.sha256(blob).hexdigest()!=p['sha256']: raise SystemExit(2)
for folder in folders: folder.mkdir(parents=True,mode=0o700)
with (folders[0]/'000000.npz').open('xb') as stream: stream.write(blob)
'''


class PublicationTransport:
    def __init__(self, source, digest, session, peer, check_failure, *, root=Path('/dev/shm/glm53-handoff')):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', session) or session in {'.', '..'}:
            raise ValueError('invalid publication session')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', peer):
            raise ValueError('invalid peer host')
        with Path(source).open('rb') as stream:
            self.payload = stream.read(1048577)
        if len(self.payload) > 1048576 or hashlib.sha256(self.payload).hexdigest() != digest:
            raise ValueError('publication size or digest differs')
        self.digest, self.session, self.peer = digest, session, peer
        self.folder = Path(root) / 'tp-live-in' / session
        self.ssh_options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
        self.delivery = RankDelivery(('local', peer), session, self.probe, 180, check_failure)
        self.staged = self.published = False

    def probe(self, host, command, *, timeout):
        argv = shlex.split(command) if host == 'local' else ['ssh', *self.ssh_options, host, command]
        result = subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, timeout=timeout)
        return result.stdout

    def _deadline(self, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise TimeoutError('publication transport deadline exceeded')
        return time.monotonic() + timeout

    def stage(self, *, timeout):
        if self.staged or self.published: raise RuntimeError('publication transport cannot be reused')
        deadline = self._deadline(timeout)
        request = json.dumps({'run': self.session, 'sha256': self.digest})
        command = f'python3 -c {shlex.quote(PEER_STAGE)} {shlex.quote(request)}'
        subprocess.run(['ssh', *self.ssh_options, self.peer, command], input=self.payload, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        if time.monotonic() >= deadline: raise TimeoutError('publication staging deadline exceeded')
        with (self.folder / '.000000.tmp.npz').open('xb') as stream:
            stream.write(self.payload)
        if time.monotonic() >= deadline: raise TimeoutError('publication staging deadline exceeded')
        self.staged = True

    def publish(self, *, timeout):
        if not self.staged or self.published: raise RuntimeError('publication transport is not staged')
        deadline = self._deadline(timeout)
        temporary, final = self.folder / '.000000.tmp.npz', self.folder / '000000.npz'
        os.link(temporary, final)
        temporary.unlink()
        self.published = True
        if time.monotonic() >= deadline: raise TimeoutError('publication release deadline exceeded')
