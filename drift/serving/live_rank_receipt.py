"""Write fixed per-rank receipts after all local cache writes have completed."""
import hashlib
import json
import os


def publication_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def acknowledge(connector, name, sequence, rows, digest):
    rank, world_size = connector._tp()
    folder = connector._live_out / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'ack.{sequence:06d}.rank{rank}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'rank': int(rank), 'world_size': int(world_size),
                                    'sequence': int(sequence), 'rows': int(rows), 'sha256': digest}))
    os.replace(temporary, path)
