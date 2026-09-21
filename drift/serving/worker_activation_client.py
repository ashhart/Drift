"""Issue bounded typed activation commands over one controller-owned Unix socket."""
import json
import re
import threading
import time


class ActivationClient:
    def __init__(self, connection, configuration, deadline):
        self.socket, self.config, self.deadline = connection, dict(configuration), deadline
        self.seq, self.lock = 0, threading.Lock()
        self.appended = self.next_first = self.selected_stop = 0

    def append(self, path, sha256, rows):
        if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256): raise ValueError('ACTIVATION_HASH')
        if self.appended + self._rows(rows) > self.config['max_total_rows']: raise ValueError('ACTIVATION_ROWS')
        return self._request('append', path, {'sha256': sha256, 'rows': rows})

    def tap(self, path, first, max_rows):
        if type(first) is not int or first != self.next_first: raise ValueError('ACTIVATION_CURSOR')
        return self._request('tap', path, {'first': first, 'max_rows': self._rows(max_rows)})

    def snapshot_last_own(self, path, after):
        if type(after) is not int or after != self.selected_stop: raise ValueError('ACTIVATION_CURSOR')
        return self._request('snapshot_last_own', path, {'after': after})

    def _rows(self, value):
        if type(value) is not int or not 1 <= value <= self.config['max_rows']: raise ValueError('ACTIVATION_ROWS')
        return value

    def _request(self, operation, path, payload):
        with self.lock:
            if type(path) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.npz', path): raise ValueError('ACTIVATION_PATH')
            if self.seq >= 1024: raise ValueError('ACTIVATION_LIMIT')
            self.seq += 1
            frame = {'v': 1, 'session': self.config['session'], 'seq': self.seq, 'op': operation, 'path': path, **payload}
            try:
                self._timeout(); self.socket.sendall(json.dumps(frame).encode() + b'\n')
                raw = bytearray()
                while b'\n' not in raw:
                    self._timeout(); block = self.socket.recv(4097 - len(raw))
                    if not block: raise ValueError('ACTIVATION_EOF')
                    raw.extend(block)
                    if len(raw) > 4096: raise ValueError('ACTIVATION_LIMIT')
                if raw.count(b'\n') != 1 or not raw.endswith(b'\n'): raise ValueError('ACTIVATION_REPLY')
                from drift.serving.worker_owner_socket import _object, _constant
                reply = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
                if operation == 'snapshot_last_own':
                    from drift.serving.worker_snapshot import validate_snapshot
                    self.selected_stop = validate_snapshot(reply, session=self.config['session'], seq=self.seq,
                        source_worker=self.config['target_worker'], target_worker=self.config['source_worker'], after=payload['after'])
                    return reply
                source, target = self.config['source_worker'], self.config['target_worker']
                if operation == 'tap': source, target = target, source
                common = {'v': 1, 'session': self.config['session'], 'seq': self.seq, 'source_worker': source, 'target_worker': target}
                required = {'op', 'rows', 'sha256', 'foreign_total' if operation == 'append' else 'next_first'}
                if type(reply) is not dict or set(reply) != set(common) | required or any(type(reply.get(key)) is not type(value) or reply.get(key) != value for key, value in common.items()): raise ValueError('ACTIVATION_REPLY')
                if reply['op'] != ('appended' if operation == 'append' else 'tapped'): raise ValueError('ACTIVATION_REPLY')
                self._rows(reply['rows'])
                if not isinstance(reply['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', reply['sha256']): raise ValueError('ACTIVATION_REPLY')
                if operation == 'append' and (reply['sha256'] != payload['sha256'] or reply['rows'] != payload['rows']): raise ValueError('ACTIVATION_REPLY')
                counter = reply['foreign_total' if operation == 'append' else 'next_first']
                if type(counter) is not int or not 0 <= counter <= self.config['max_total_rows']: raise ValueError('ACTIVATION_REPLY')
                if operation == 'tap' and (reply['rows'] > payload['max_rows'] or counter != payload['first'] + reply['rows']): raise ValueError('ACTIVATION_REPLY')
                if operation == 'append' and counter != self.appended + payload['rows']: raise ValueError('ACTIVATION_REPLY')
                if operation == 'append': self.appended = counter
                else: self.next_first = counter
                return reply
            except Exception:
                self.close()
                raise RuntimeError('ACTIVATION_CHANNEL_FAILED') from None

    def _timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError('ACTIVATION_DEADLINE')
        self.socket.settimeout(min(remaining, 2))

    def close(self):
        self.socket.close()
