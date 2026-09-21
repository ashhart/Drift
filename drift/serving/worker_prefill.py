"""Apply one admitted own-input suffix in bounded native calls under a single cache lock."""
from contextlib import nullcontext
import time
from drift.serving.worker_contract import require


def prefill_chunks(handle, state, ids, cancelled, deadline):
    transaction = handle.transaction() if hasattr(handle, 'transaction') else nullcontext()
    try:
        with transaction:
            for offset in range(0, len(ids), 4096):
                require(not cancelled.is_set() and not state.get('poisoned'), 'WORKER')
                require(time.monotonic() < deadline, 'LIMIT')
                chunk = ids[offset:offset + 4096]
                handle({'op': 'continue', 'ids': chunk})
                yield len(chunk)
                require(not cancelled.is_set(), 'WORKER')
                require(time.monotonic() < deadline, 'LIMIT')
    except Exception:
        state['poisoned'] = True
        if hasattr(handle, 'poison'):
            handle.poison()
        raise
