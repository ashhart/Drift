"""Handle separately authorized activation controls without exposing own-input text."""
import math
import os
from drift.serving.worker_contract import fields, integer, require
from drift.serving.worker_native_gate import NativeGate
from drift.serving.worker_activation_files import bounded_path, checksum, private_root, publication, snapshot, temporary


class ActivationControl:
    def __init__(self, config, gate, layouts):
        self.root = private_root(config['memory_root'])
        self.session, self.gate, self.layouts = config['session'], gate, layouts
        require(type(self.session) is str and 0 < len(self.session) <= 96)
        self.source, self.target = config['source_worker'], config['target_worker']
        require(all(type(value) is str and 0 < len(value) <= 96 for value in (self.source, self.target)))
        require(self.source != self.target)
        self.max_bytes = integer(config['max_bytes'], 1, 128 * 1024 * 1024)
        self.max_rows = integer(config['max_rows'], 1, 4096)
        self.max_total = integer(config['max_total_rows'], 1, 1_000_000)
        self.seq = self.appended = self.next_first = self.selected_stop = 0

    def apply(self, frame):
        try:
            require(type(frame) is dict)
            operation = frame.get('op')
            extra = ('sha256', 'rows') if operation == 'append' else ('after',) if operation == 'snapshot_last_own' else ('first', 'max_rows')
            fields(frame, ('v', 'session', 'seq', 'op', 'path', *extra))
            require(type(frame['v']) is int and frame['v'] == 1 and frame['session'] == self.session)
            require(type(frame['seq']) is int and frame['seq'] == self.seq + 1)
            require(operation in ('append', 'tap', 'snapshot_last_own'))
            path = bounded_path(self.root, frame['path'])
            if operation == 'snapshot_last_own':
                from drift.serving.worker_snapshot_export import snapshot_last_own
                result = snapshot_last_own(self, frame, path)
            else:
                result = self._append(frame, path) if operation == 'append' else self._tap(frame, path)
            self.seq = frame['seq']
            source, target = (self.source, self.target) if operation == 'append' else (self.target, self.source)
            return {'v': 1, 'session': self.session, 'seq': self.seq, 'source_worker': source, 'target_worker': target, **result}
        except Exception:
            self.gate.poison()
            raise RuntimeError('ACTIVATION_CONTROL_FAILED') from None

    def _append(self, frame, path):
        rows = integer(frame['rows'], 1, self.max_rows)
        require(self.appended + rows <= self.max_total, 'LIMIT')
        target = temporary(self.root)
        try:
            digest = snapshot(path, target, self.max_bytes)
            require(digest == frame['sha256'], 'CAPABILITY')
            require(publication(target, self.layouts, self.max_bytes, self.max_rows) == rows)
            with self.gate.transaction():
                result = self.gate({'op': 'append', 'memory': str(target)})
                require(result['appended'] == rows)
                self.appended += rows
                self.gate.state['activation_receipt'] = {'session': self.session, 'target_worker': self.target,
                                                       'activation_seq': frame['seq'], 'foreign_total': self.appended}
            return {'op': 'appended', 'rows': rows, 'sha256': digest, 'foreign_total': self.appended}
        finally:
            target.unlink(missing_ok=True)

    def _tap(self, frame, path):
        first = integer(frame['first'], 0, self.max_total)
        maximum = integer(frame['max_rows'], 1, self.max_rows)
        require(first == self.next_first and not path.exists())
        target = temporary(self.root)
        try:
            with self.gate.transaction():
                rows = len(self.gate.state['own_slots']) - first
                require(0 < rows <= maximum and first + rows <= self.max_total, 'LIMIT')
                estimated = 4096 + sum(rows * math.prod(shape) * 2 + 1024 for shape in self.layouts.values())
                require(estimated <= self.max_bytes, 'LIMIT')
                result = self.gate({'op': 'tap', 'first': first, 'out': str(target)})
                require(result['tapped'] == rows and result['next_first'] == first + rows)
                require(publication(target, self.layouts, self.max_bytes, maximum) == rows)
                digest = checksum(target, self.max_bytes)
                os.link(target, path)
                self.next_first = first + rows
            return {'op': 'tapped', 'rows': rows, 'next_first': self.next_first, 'sha256': digest}
        finally:
            target.unlink(missing_ok=True)
