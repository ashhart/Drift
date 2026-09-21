"""Own bounded immutable activation versions independently of native worker input."""
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import threading
import uuid
from drift.serving.worker_activation_files import bounded_path, checksum, private_root, publication, snapshot, temporary


def require(condition):
    if not condition: raise ValueError('snapshot bank contract mismatch')


def bounded(value, maximum):
    require(type(value) is int and 0 < value <= maximum)
    return value


@dataclass(frozen=True)
class Snapshot:
    session: str
    version: int
    path: str
    sha256: str
    rows: int
    bytes: int
    source_worker: str
    target_worker: str
    recipe_sha256: str

    def receipt(self):
        return {key: value for key, value in asdict(self).items() if key != 'path'}


class SnapshotBank:
    def __init__(self, root, *, session, source_worker, target_worker, recipe_sha256, layouts, max_rows, max_bytes, max_versions, max_total_bytes):
        self.root = private_root(root)
        for label in (session, source_worker, target_worker):
            require(type(label) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', label))
        require(source_worker != target_worker and type(recipe_sha256) is str and re.fullmatch('[0-9a-f]{64}', recipe_sha256))
        require(type(layouts) is dict and 0 < len(layouts) <= 256)
        require(all(re.fullmatch(r'l[0-9]+', key) and tuple(value) == (512,) for key, value in layouts.items()))
        self.session, self.source, self.target, self.recipe = session, source_worker, target_worker, recipe_sha256
        self.layouts = {key: tuple(value) for key, value in layouts.items()}
        self.max_rows, self.max_bytes = bounded(max_rows, 4096), bounded(max_bytes, 1048576)
        self.max_versions, self.max_total = bounded(max_versions, 1024), bounded(max_total_bytes, 128 * 1048576)
        self.directory = self.root / ('bank-' + uuid.uuid4().hex); self.directory.mkdir(mode=0o700)
        self.lock, self.failed, self.versions, self.total = threading.RLock(), False, [], 0

    def publish(self, frame):
        with self.lock:
            target = None
            try:
                require(not self.failed and type(frame) is dict)
                frame = dict(frame)
                require(set(frame) == {'v', 'session', 'version', 'path', 'sha256', 'rows', 'source_worker', 'target_worker', 'recipe_sha256'})
                require(type(frame['v']) is int and frame['v'] == 1 and frame['session'] == self.session)
                require(type(frame['version']) is int and frame['version'] == len(self.versions) + 1 <= self.max_versions)
                require((frame['source_worker'], frame['target_worker'], frame['recipe_sha256']) == (self.source, self.target, self.recipe))
                rows = bounded(frame['rows'], self.max_rows)
                source = bounded_path(self.root, frame['path']); target = temporary(self.directory)
                value = snapshot(source, target, self.max_bytes)
                require(value == frame['sha256'] and publication(target, self.layouts, self.max_bytes, self.max_rows) == rows)
                size = target.stat().st_size; require(self.total + size <= self.max_total)
                final = self.directory / f'version-{frame["version"]:06d}.npz'
                target.chmod(0o400); os.link(target, final)
                captured = Snapshot(self.session, frame['version'], str(final), value, rows, size, self.source, self.target, self.recipe)
                self.versions.append(captured); self.total += size
                return captured.receipt()
            except Exception:
                self.failed = True
                raise ValueError('SNAPSHOT_PUBLICATION_FAILED') from None
            finally:
                if target is not None: target.unlink(missing_ok=True)

    def validate(self, captured):
        with self.lock:
            try:
                require(not self.failed and isinstance(captured, Snapshot) and any(captured is value for value in self.versions))
                path = Path(captured.path)
                require(path.resolve() == path and not path.is_symlink() and path.stat().st_size == captured.bytes)
                require(path.stat().st_mode & 0o777 == 0o400 and checksum(path, self.max_bytes) == captured.sha256)
            except Exception:
                self.failed = True
                raise ValueError('SNAPSHOT_CAPTURE_FAILED') from None
            return captured

    def capture(self):
        with self.lock:
            require(not self.failed and bool(self.versions))
            return self.validate(self.versions[-1])

    def close(self):
        with self.lock: self.failed = True
