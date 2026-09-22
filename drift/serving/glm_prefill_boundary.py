"""Bound initial prefill so borrowed memory precedes the remaining own input."""


def boundary(params, alignment=64):
    value = params.get('drift_prefill_boundary')
    if value is None:
        return None
    start, rows = params.get('drift_reserve_start'), params.get('drift_reserve')
    owner = bool(params.get('drift_session')) != (params.get('drift_no_link') is True)
    if (not owner or type(alignment) is not int or alignment != 64
            or any(type(x) is not int for x in (value, start, rows))
            or not 0 <= start < start+rows < value <= 65536
            or not 1 <= rows <= 4096 or value-start > 4096 or value % alignment):
        raise ValueError('invalid causal prefill boundary')
    return value


def cap_prefill(request, count, start, alignment):
    stop = boundary(getattr(request, 'kv_transfer_params', None) or {}, alignment)
    if stop is None:
        return count
    if any(type(x) is not int or x < 0 for x in (count, start)):
        raise ValueError('invalid causal prefill counters')
    return min(count, stop-start) if start < stop else count


def guard_initial(state, before, after, ready):
    stop = state.get('prefill_boundary')
    if stop is None or state['failed'] or state['next_seq']:
        return
    if before < 0 or after > stop or (after >= state['start']+state['reserve'] and not ready):
        state['failed'] = 'initial memory must apply before the own-input boundary'
