"""Verify explicit worker manifests and their listed artifacts before backend dispatch."""
import hashlib
import json
from pathlib import Path
import re
from drift.serving.worker_contract import require


def digest(path, maximum):
    value = hashlib.sha256()
    consumed = 0
    with Path(path).open('rb') as source:
        while chunk := source.read(min(1024 * 1024, maximum - consumed + 1)):
            consumed += len(chunk)
            require(consumed <= maximum, 'LIMIT')
            value.update(chunk)
    return value.hexdigest()


def verify_worker_manifests(configuration):
    receipts = {}
    maximum = configuration.get('max_verified_artifact_bytes', 64 * 1024 * 1024)
    require(type(maximum) is int and maximum > 0, 'LIMIT')
    total = 0
    for kind in ('model', 'translator'):
        path = Path(configuration[kind + '_manifest_path'])
        require(path.absolute() == path.resolve(), 'CAPABILITY')
        expected = configuration['pins'][kind + '_sha256']
        require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None, 'CAPABILITY')
        require(path.stat().st_size <= 1024 * 1024, 'LIMIT')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'CAPABILITY')
        spec = json.loads(raw)
        if kind == 'model':
            require(type(spec.get('weights_verification')) is str and bool(spec['weights_verification']), 'CAPABILITY')
        artifacts, hashes = spec['artifacts'], spec['sha256']
        require(type(artifacts) is dict and type(hashes) is dict and 0 < len(artifacts) <= 1024 and set(artifacts) == set(hashes), 'CAPABILITY')
        for name, artifact in artifacts.items():
            require(type(artifact) is str and type(hashes[name]) is str and re.fullmatch('[0-9a-f]{64}', hashes[name]) is not None, 'CAPABILITY')
            artifact_path = Path(artifact)
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            require(artifact_path.absolute() == artifact_path.resolve(), 'CAPABILITY')
            length = artifact_path.stat().st_size
            require(total + length <= maximum, 'LIMIT')
            require(digest(artifact_path, maximum - total) == hashes[name], 'CAPABILITY')
            total += length
        require(digest(path, 1024 * 1024) == expected, 'CAPABILITY')
        receipts[kind] = {'manifest_sha256': expected, 'verified_artifacts': len(artifacts)}
        if kind == 'model':
            receipts[kind].update(weights_verification=spec['weights_verification'], checkpoint_path=spec.get('checkpoint_path'))
    return receipts
