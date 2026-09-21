"""Finalize only accepted cached rows from a private worker-side tail snapshot."""
import re
from zipfile import ZipFile

import numpy as np

try:
    from live_publication import load_publication, wire_arrays
    from live_tap_capture import save_arrays
except ImportError:
    from drift.serving.live_publication import load_publication, wire_arrays
    from drift.serving.live_tap_capture import save_arrays


def final_frontier(request):
    if getattr(getattr(request, 'status', None), 'name', None) not in ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED'):
        raise ValueError('live tap request did not finish successfully')
    names = ('num_computed_tokens', 'num_tokens', 'num_in_flight_tokens', 'num_stale_output_tokens')
    values = [getattr(request, name, None) for name in names]
    if any(type(value) is not int or value < 0 for value in values) or values[2] or values[3]:
        raise ValueError('live tap terminal frontier is not settled')
    return min(values[:2])


def finish_taps(folder, state, request):
    source_start = state['start'] + state['reserve']
    final_stop = final_frontier(request)
    path = folder / '.pending' / 'tail.npz'
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) > 132:
            raise ValueError('live tap layer count outside bound')
        layers = {name[:-4]: (512,) for name in names if re.fullmatch(r'l[0-9]+\.npy', name)}
    if not layers:
        raise ValueError('live tap snapshot has no layers')
    arrays = load_publication(path, layers, metadata=('start', 'stop', 'verified', 'sequence'))
    start, stop, verified, sequence = (arrays.pop(name) for name in ('start', 'stop', 'verified', 'sequence'))
    if not source_start <= start <= final_stop <= stop or not verified <= final_stop:
        raise ValueError('live tap snapshot does not cover accepted frontier')
    if any(value.shape[0] != stop - start for value in arrays.values()):
        raise ValueError('live tap snapshot row count mismatch')
    published = sorted(folder.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz'))
    if len(published) != sequence or sequence >= 10**6 or [path.name for path in published] != [f'{index:06d}.npz' for index in range(sequence)]:
        raise ValueError('live tap snapshot sequence mismatch')
    previous_stop = source_start
    for path in published:
        previous = load_publication(path, layers, metadata=('start', 'stop'))
        previous_start = previous.pop('start')
        if previous_start != previous_stop:
            raise ValueError('live tap publication positions are not contiguous')
        previous_stop = previous.pop('stop')
        if previous_stop <= previous_start or any(value.shape[0] != previous_stop - previous_start for value in previous.values()):
            raise ValueError('live tap publication row count mismatch')
    if previous_stop != start:
        raise ValueError('live tap snapshot does not follow publications')
    if final_stop > start:
        tail = wire_arrays({name: value[:final_stop - start] for name, value in arrays.items()}, layers, final_stop - start)
        save_arrays(folder / f'{sequence:06d}.npz', start=np.array(start), stop=np.array(final_stop), **tail)
        sequence += 1
    return {'tap_count': sequence, 'source_start': source_start, 'source_stop': final_stop}
