"""Bound initial prefill so borrowed memory precedes the remaining own input."""


def boundary(params):
    """The position where a linked request's initial prefill stops, or None. It may equal the reserve's end."""
    value = params.get('drift_prefill_boundary')
    if value is None:
        return None
    start, rows = params.get('drift_reserve_start'), params.get('drift_reserve')
    owner = bool(params.get('drift_session')) != (params.get('drift_no_link') is True)
    if (not owner or any(type(x) is not int for x in (value, start, rows))
            or not 0 <= start < start+rows <= value <= 65536
            or not 1 <= rows <= 4096 or value-start > 4096):
        raise ValueError('invalid causal prefill boundary')
    return value


def split(request, count, start, block_size, aligned):
    """Tokens for this prefill chunk; `aligned(n)` is vLLM's block-aligned split of n tokens.

    A chunk that reaches a boundary inside its current block ends exactly there and keeps a private
    running state, as vLLM's own sub-block chunks do. Before that, vLLM aligns the chunk capped at the boundary.
    """
    stop = boundary(getattr(request, 'kv_transfer_params', None) or {})
    if stop is None or start >= stop:
        return aligned(count)
    if any(type(x) is not int or x < 0 for x in (count, start)) or type(block_size) is not int or block_size < 1:
        raise ValueError('invalid causal prefill counters')
    capped = min(count, stop-start)
    if start+capped == stop and stop <= (start//block_size+1)*block_size:
        return capped
    return aligned(capped)


def guard_initial(state, before, after, ready):
    stop = state.get('prefill_boundary')
    if stop is None or state['failed'] or state['next_seq']:
        return
    if before < 0 or after > stop or (after >= state['start']+state['reserve'] and not ready):
        state['failed'] = 'initial memory must apply before the own-input boundary'
