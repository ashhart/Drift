"""Expose only completed application receipts, never native prompts or answers."""
import copy

from drift.serving.glm_snapshot_bank import require


def applied_snapshot(bank, restorer, version, digest):
    require(type(version) is int and 1 <= version <= len(bank.versions))
    captured = bank.validate(bank.versions[version-1])
    require(captured.sha256 == digest)
    result = dict(op='application', state='PENDING', version=version,
                  sha256=digest, rows=captured.rows, receipts=[])
    with restorer.boundary:
        require(not restorer.failed)
        turns = [turn for turn in restorer.completed if turn['snapshot']['version'] == version]
        if not turns:
            return result
        turn = turns[-1]
        require(turn['snapshot_sha256'] == digest and turn['snapshot']['sha256'] == digest)
        receipts = turn['receipts']
        require(type(receipts) is list and len(receipts) == 2)
        require(sorted(receipt['rank'] for receipt in receipts) == [0, 1])
        for receipt in receipts:
            require(receipt == dict(rank=receipt['rank'], world_size=2, sequence=0,
                                    rows=captured.rows, sha256=digest))
            require(type(receipt['rank']) is int and type(receipt['world_size']) is int
                    and type(receipt['rows']) is int and type(receipt['sequence']) is int)
        return {**result, 'state': 'APPLIED', 'receipts': copy.deepcopy(receipts)}
