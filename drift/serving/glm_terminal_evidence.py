"""Validate accepted-cache terminal evidence against contiguous source receipts."""
import re


VERIFIED = 'VERIFIED_ACCEPTED_CACHE'
TERMINAL_FIELDS = {'tap_count', 'source_start', 'source_stop'}


def require(condition):
    if not condition:
        raise ValueError('native terminal evidence mismatch')


def terminal_fields(marker):
    legacy = {'failed', 'writes_scheduled'}
    require(type(marker) is dict and set(marker) in (legacy, legacy | TERMINAL_FIELDS))
    require(marker['failed'] == '' and type(marker['writes_scheduled']) is int and marker['writes_scheduled'] >= 0)
    if set(marker) == legacy:
        return None
    result = {key: marker[key] for key in TERMINAL_FIELDS}
    require(all(type(value) is int and value >= 0 for value in result.values()))
    require(result['source_start'] <= result['source_stop'] and result['tap_count'] <= 1024)
    return result


def verify_terminal(terminal, raw, start, stop, raw_bytes, selected):
    require(type(terminal) is dict and set(terminal) == TERMINAL_FIELDS)
    require(all(type(value) is int and value >= 0 for value in terminal.values()))
    require(type(raw) is list and len(raw) == terminal['tap_count'] <= 1024)
    require(terminal['source_start'] == start and terminal['source_stop'] == stop)
    cursor, size = start, 0
    for seq, item in enumerate(raw):
        require(type(item) is dict and set(item) == {'seq', 'sha256', 'bytes', 'start', 'stop'})
        require(type(item['seq']) is int and item['seq'] == seq)
        require(type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']))
        require(all(type(item[key]) is int for key in ('bytes', 'start', 'stop')))
        require(item['bytes'] > 0 and item['start'] == cursor < item['stop'])
        cursor, size = item['stop'], size + item['bytes']
    require(cursor == stop and size == raw_bytes)
    for item in selected:
        seq = item['source_seq']
        require(type(seq) is int and 0 <= seq < len(raw))
        source = raw[seq]
        require(source['start'] <= item['source_start'] < item['source_stop'] == source['stop'])
