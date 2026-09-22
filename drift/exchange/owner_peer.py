"""Keep one bounded private control connection to a native owner."""
import json
import os
from pathlib import Path
import socket
import stat
import time

from drift.exchange.lifetime import checkpoint
from drift.serving.worker_owner_socket import _object, _constant


class OwnerPeer:
    def __init__(self, path, deadline):
        path = Path(path)
        info = path.lstat()
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or path.parent.resolve() != path.parent or stat.S_IMODE(path.parent.stat().st_mode) != 0o700):
            raise ValueError('OWNER_PEER_PATH')
        self.deadline, self.requests, self.closed = deadline, 0, False
        self.socket = socket.socket(socket.AF_UNIX)
        try:
            self._timeout(); self.socket.connect(str(path))
        except Exception:
            self.close(); raise

    def _timeout(self):
        checkpoint()
        remaining = self.deadline-time.monotonic()
        if self.closed or remaining <= 0:
            raise TimeoutError('OWNER_PEER_DEADLINE')
        self.socket.settimeout(min(.1, remaining))

    def request(self, frame):
        try:
            raw = json.dumps(frame, allow_nan=False).encode()+b'\n'
            if len(raw) > 4096 or self.requests >= 1024:
                raise ValueError('OWNER_PEER_LIMIT')
            self._timeout(); self.socket.sendall(raw)
            reply = bytearray()
            while b'\n' not in reply:
                self._timeout()
                try: part = self.socket.recv(4097-len(reply))
                except socket.timeout: continue
                if not part: raise EOFError('OWNER_PEER_EOF')
                reply.extend(part)
                if len(reply) > 4096: raise ValueError('OWNER_PEER_LIMIT')
            if reply.count(b'\n') != 1 or not reply.endswith(b'\n'):
                raise ValueError('OWNER_PEER_FRAME')
            value = json.loads(reply, object_pairs_hook=_object, parse_constant=_constant)
            self.requests += 1
            return value
        except Exception:
            self.close(); raise

    def close(self):
        self.closed = True
        self.socket.close()
