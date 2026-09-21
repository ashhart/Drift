"""Validate one bounded snapshot exchange without implying semantic transfer."""
import re

MODES = ('drift', 'text')
MAX_ROWS = 1_000_000
MAX_BYTES = 1 << 30
FIELDS = ('v', 'session', 'sequence', 'mode', 'source_worker', 'target_worker', 'rows', 'source_rows',
          'copies', 'bytes', 'source_sha256', 'text_bytes', 'applied_ranks', 'receipts')
DIGEST = re.compile('[0-9a-f]{64}')


def require(condition, message='EXCHANGE_CONTRACT'):
    if not condition:
        raise ValueError(message)


def integer(value, low, high, message='EXCHANGE_RANGE'):
    require(type(value) is int and low <= value <= high, message)
    return value


def digest(value, message='EXCHANGE_DIGEST'):
    require(type(value) is str and DIGEST.fullmatch(value) is not None, message)
    return value


def validate_exchange(record, *, session, sequence, mode, source_worker, target_worker, expected_ranks):
    """Return the row count this record evidences, or raise; a text-mode record evidences zero rows."""
    require(type(record) is dict and tuple(sorted(record)) == tuple(sorted(FIELDS)), 'EXCHANGE_FIELDS')
    require(mode in MODES and record['mode'] == mode, 'EXCHANGE_MODE')
    require(source_worker != target_worker, 'EXCHANGE_ROUTE')
    bound = dict(v=1, session=session, sequence=sequence, source_worker=source_worker, target_worker=target_worker)
    require(all(type(record[key]) is type(value) and record[key] == value for key, value in bound.items()), 'EXCHANGE_BINDING')
    integer(record['sequence'], 0, MAX_ROWS)
    integer(record['text_bytes'], 0, MAX_BYTES)
    require(type(record['applied_ranks']) is tuple and type(record['receipts']) is tuple, 'EXCHANGE_RANKS')
    if mode == 'text':
        return _text(record)
    return _drift(record, tuple(expected_ranks))


def _text(record):
    """A text-mode arm is the no-link baseline: it must carry no rows, no publication and no receipts."""
    require(record['rows'] == 0 and record['source_rows'] == 0 and record['copies'] == 0, 'EXCHANGE_TEXT_ROWS')
    require(record['source_sha256'] is None and record['bytes'] == 0, 'EXCHANGE_TEXT_BODY')
    require(record['applied_ranks'] == () and record['receipts'] == (), 'EXCHANGE_TEXT_RANKS')
    return 0


def _drift(record, expected_ranks):
    """A drift-mode arm must evidence tiled rows applied at every expected rank, with no text crossing."""
    require(record['text_bytes'] == 0, 'EXCHANGE_TEXT_FALLBACK')
    source_rows = integer(record['source_rows'], 1, MAX_ROWS)
    copies = integer(record['copies'], 1, MAX_ROWS)
    rows = integer(record['rows'], 1, MAX_ROWS)
    require(rows == source_rows * copies, 'EXCHANGE_ROWS')
    integer(record['bytes'], 1, MAX_BYTES)
    digest(record['source_sha256'])
    require(record['applied_ranks'] == expected_ranks, 'EXCHANGE_RANKS')
    require(len(record['receipts']) == len(expected_ranks), 'EXCHANGE_RECEIPTS')
    for value in record['receipts']:
        digest(value, 'EXCHANGE_RECEIPTS')
    return rows
