"""Export only the latest own slot while holding the native mutation transaction."""
import math
import os
from drift.serving.worker_contract import integer, require
from drift.serving.worker_activation_files import checksum, publication, temporary
from drift.serving.worker_snapshot import MAX_OWN_ROWS


def snapshot_last_own(control, frame, path):
    after = integer(frame['after'], 0, MAX_OWN_ROWS)
    require(after == control.selected_stop and not path.exists())
    target = temporary(control.root)
    try:
        with control.gate.transaction():
            state = control.gate.state
            count = integer(len(state['own_slots']), 1, MAX_OWN_ROWS)
            require(count > after)
            slot = integer(state['own_slots'][-1], 0, len(state['slot_pos']) - 1)
            position = integer(state['slot_pos'][slot], state['reserve'], 2**31 - 1)
            require(position == state['own_pos'] - 1)
            require(not state.get('span') or not state['span'][0] <= position < state['span'][1])
            estimated = 4096 + sum(math.prod(shape) * 2 + 1024 for shape in control.layouts.values())
            require(estimated <= control.max_bytes, 'LIMIT')
            result = control.gate({'op': 'tap', 'first': count - 1, 'out': str(target)})
            require(type(result.get('tapped')) is int and result['tapped'] == 1)
            require(type(result.get('next_first')) is int and result['next_first'] == count)
            require(len(state['own_slots']) == count and state['own_slots'][-1] == slot)
            require(publication(target, control.layouts, control.max_bytes, 1) == 1)
            digest = checksum(target, control.max_bytes)
            target.chmod(0o400); os.link(target, path)
            control.selected_stop = count
        return dict(op='own_snapshot', selection='last_own_row', scope='own', rows=1, after=after,
                    source_start=count - 1, source_stop=count, available_own_rows=count, sha256=digest)
    finally:
        target.unlink(missing_ok=True)
