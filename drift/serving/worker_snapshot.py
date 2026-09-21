"""Validate one explicitly selected own-row snapshot without implying prefix delivery."""
import re
from drift.serving.worker_contract import fields, integer, require

MAX_OWN_ROWS = 1_000_000


def validate_snapshot(receipt, *, session, seq, source_worker, target_worker, after):
    integer(after, 0, MAX_OWN_ROWS)
    expected = dict(v=1, session=session, seq=seq, op='own_snapshot', source_worker=source_worker,
                    target_worker=target_worker, selection='last_own_row', scope='own', rows=1, after=after)
    fields(receipt, (*expected, 'source_start', 'source_stop', 'available_own_rows', 'sha256'))
    require(source_worker != target_worker)
    require(all(type(receipt[key]) is type(value) and receipt[key] == value for key, value in expected.items()))
    count = integer(receipt['available_own_rows'], 1, MAX_OWN_ROWS)
    require(count > after)
    require(type(receipt['source_start']) is int and receipt['source_start'] == count - 1)
    require(type(receipt['source_stop']) is int and receipt['source_stop'] == count)
    require(type(receipt['sha256']) is str and re.fullmatch('[0-9a-f]{64}', receipt['sha256']) is not None)
    return count
