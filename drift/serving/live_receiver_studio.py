"""Validate a complete incoming Studio memory before mutating a running cache."""
import numpy as np

from drift.serving.live_publication import count, load_publication


def append_memory(state, path, layouts, rope, index_dim, append, dtype, settle):
    """A failed append poisons the session; only a fresh start may recover it."""
    if state.get('poisoned'):
        raise RuntimeError('live session is poisoned; start a fresh session')
    state['poisoned'] = True
    wire_layouts = {f'{kind}{layer}': shape for layer, shape in layouts.items() for kind in 'kv'}
    arrays = load_publication(path, wire_layouts)
    entries = {layer: (arrays[f'k{layer}'], arrays[f'v{layer}']) for layer in layouts}
    if any((np.abs(value) > np.finfo(np.float16).max).any() for value in arrays.values()):
        raise ValueError('incoming memory exceeds the float16 wire range')
    rows = next(iter(entries.values()))[0].shape[0]
    offsets = set()
    for layer, shape in layouts.items():
        cache = state['cache'][layer]
        offsets.add(count(cache.offset))
        for tensor in (cache.keys, cache.values):
            if tensor is not None and (tensor.ndim != 4 or tensor.shape[0] != 1 or (tensor.shape[1], tensor.shape[3]) != shape):
                raise ValueError('receiver cache shape disagrees with declared layout')
        if hasattr(cache, 'update_indexer'):
            count(index_dim, positive=True)
    if len(offsets) != 1 or offsets != {len(state['slot_pos'])}:
        raise ValueError('receiver KV layers disagree about cache length')
    if state.get('span'):
        start = count(state['span_next'])
        if start + rows > state['span'][1]:
            raise ValueError('reserved span exhausted')
    else:
        top = count(state['reserve']) - count(state['foreign_pos'])
        if rows > top:
            raise ValueError('reserved foreign positions exhausted')
        start = top - rows
    positions = np.arange(start, start + rows)
    append(state['cache'], entries, rope, positions, index_dim, dtype=dtype)
    tensors = [getattr(state['cache'][layer], name, None) for layer in layouts for name in ('keys', 'values', 'index_keys')]
    settle(*[tensor for tensor in tensors if tensor is not None])
    state['slot_pos'] += positions.tolist()
    state['foreign_pos'] += rows
    if state.get('span'):
        state['span_next'] += rows
    state['poisoned'] = False
    return {'appended': rows, 'foreign_total': state['foreign_pos'], 'cache_slots': len(state['slot_pos'])}
