"""Retain Studio native cache and boundary tokens across explicit own-input turns."""


def continue_native(state, ids, run, max_tokens=4096):
    if state.get('poisoned') or 'cache' not in state:
        raise RuntimeError('native session unavailable')
    if type(ids) is not list or not 0 < len(ids) <= max_tokens or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('invalid continuation token IDs')
    pending = state.get('pending_stop')
    if pending is not None and (type(pending) is not int or pending < 0):
        state['poisoned'] = True
        raise RuntimeError('invalid native boundary')
    try:
        run(([pending] if pending is not None else []) + ids)
        state['pending_stop'] = None
        state['done'] = False
    except Exception:
        state['poisoned'] = True
        raise RuntimeError('native continuation failed') from None
    return {'tokens': len(ids), 'boundary_tokens': int(pending is not None), 'own_slots': len(state.get('own_slots', []))}


def generate_native(state, tokens, stops, run, decode, include_ids=False):
    if state.get('poisoned') or 'cache' not in state:
        raise RuntimeError('native session unavailable')
    if type(tokens) is not int or not 0 <= tokens <= 4096:
        raise ValueError('invalid generation budget')
    out = []
    for _ in range(tokens):
        if state['done']:
            break
        nxt = int(state['last'].argmax())
        if nxt in stops:
            state['pending_stop'] = nxt
            state['done'] = True
            break
        out.append(nxt)
        run([nxt])
    result = {'text': decode(out), 'tokens': len(out), 'done': state['done']}
    if include_ids:
        result.update(ids=out, stop_id=state.get('pending_stop'))
    return result
