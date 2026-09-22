"""Translate owner-selected snapshots and verified MCDMA collector publications."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from drift.exchange.owner_links import require
from drift.serving.translation_bridge import translate_own_snapshot, translate_prefix


def bounded_bytes(path):
    with Path(path).open('rb') as stream:
        body = stream.read(1048577)
    require(0 < len(body) <= 1048576)
    return body


def payload(path, rows):
    body = bounded_bytes(path)
    return dict(body=body, rows=rows, sha256=hashlib.sha256(body).hexdigest())


class SelectedOwnerSource:
    def __init__(self, peer, root, output, recipe, *, session, source_worker, target_worker):
        self.peer, self.root, self.output, self.recipe = peer, Path(root), Path(output), recipe
        self.session, self.source, self.target = session, source_worker, target_worker
        self.after, self.reports = 0, []

    def snapshot(self, copies):
        require(copies == 1)
        name = f'selected-{len(self.reports):06d}.npz'
        receipt = self.peer.request(dict(op='snapshot_last_own', path=name, after=self.after))
        result = translate_own_snapshot(self.recipe, self.root/name, receipt, session=self.session,
            source_worker=self.source, target_worker=self.target, seq=self.peer.requests, after=self.after,
            output=self.output/name, max_source_bytes=1048576, max_output_rows=12, max_output_bytes=1048576)
        self.after = result['source_stop']; self.reports.append(result)
        return payload(result['path'], result['rows'])


class CollectedOwnerSource:
    def __init__(self, exports, output, recipe, *, source_worker, target_worker):
        self.exports, self.output, self.recipe = Path(exports), Path(output), recipe
        self.source, self.target, self.seen, self.reports = source_worker, target_worker, set(), []

    def snapshot(self, copies):
        require(copies == 1)
        paths = sorted(self.exports.glob('*/manifest.json'), key=lambda path:path.stat().st_mtime_ns)
        require(bool(paths) and str(paths[-1]) not in self.seen)
        path = paths[-1]; raw = bounded_bytes(path)
        report = json.loads(raw); parts, reports = {}, []
        require(0 < len(report['publications']) <= 16)
        for seq in range(len(report['publications'])):
            item = translate_prefix(self.recipe, path, hashlib.sha256(raw).hexdigest(), publication_seq=seq,
                session=report['session'], source_worker=self.source, target_worker=self.target,
                output=self.output/f'forward-{len(self.reports):06d}-{seq:02d}.npz', max_source_rows=64,
                max_source_bytes=1048576, max_output_rows=128, max_output_bytes=1048576, remaining=128)
            with np.load(item['path'], allow_pickle=False) as arrays:
                for key in arrays.files: parts.setdefault(key, []).append(arrays[key])
            reports.append(item)
        rows = sum(item['rows'] for item in reports); require(0 < rows <= 128)
        stream = io.BytesIO(); np.savez(stream, **{key:np.concatenate(value) for key,value in parts.items()})
        body = stream.getvalue(); require(len(body) <= 1048576)
        self.seen.add(str(path)); self.reports.append(reports)
        return dict(body=body, rows=rows, sha256=hashlib.sha256(body).hexdigest())
