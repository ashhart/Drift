"""Read a one-row own snapshot whose absolute interval is bound to its receipt."""
from drift.serving.bridge_files import QWEN_LAYOUT, integer
from drift.serving.bridge_sources import actors, read_arrays
from drift.serving.worker_snapshot import validate_snapshot


def own_snapshot_source(path, receipt, *, session, source_worker, target_worker, seq, after, scratch, max_bytes):
    actors(session, source_worker, target_worker); integer(seq, 1024)
    count = validate_snapshot(receipt, session=session, seq=seq, source_worker=source_worker,
                              target_worker=target_worker, after=after)
    arrays = read_arrays(path, receipt['sha256'], QWEN_LAYOUT, scratch, 1, max_bytes, 1)
    return arrays, dict(source_session=session, source_seq=seq, source_sha256=receipt['sha256'],
                        source_start=count - 1, source_stop=count, source_rows=1, available_source_rows=count,
                        transmitted_source_rows=1, source_after=after, source_selection='last_own_row', scope='own')
