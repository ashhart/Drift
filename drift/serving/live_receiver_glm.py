"""Admission, write staging and poison state for the GLM live receiver."""
import os
import traceback

import numpy as np

try:
    from live_tap_capture import capture_taps
except ImportError:
    from drift.serving.live_tap_capture import capture_taps

try:
    from live_publication import load_publication
    from live_rank_receipt import acknowledge, publication_digest
    from glm53_handoff import pack_fp8_ds_mla
except ImportError:
    from drift.serving.live_publication import load_publication
    from drift.serving.live_rank_receipt import acknowledge, publication_digest
    from drift.serving.glm53_handoff import pack_fp8_ds_mla

LATENT, PACKED = 512, 528


class LiveReceiverError(RuntimeError):
    """Carry only allowlisted numeric receiver diagnostics."""
    fields = {'PAGE_RANGE': {'start', 'stop', 'pages', 'slots'}, 'BLOCK_GROUP': {'group', 'groups'},
              'PREFIX_HIT': {'hit', 'start'}, 'PREEMPTED': {'computed'}, 'NO_STORE': set()}

    def __init__(self, code, **details):
        if code not in self.fields or set(details) != self.fields[code]:
            raise ValueError('invalid receiver diagnostic fields')
        if any(type(value) is not int for value in details.values()):
            raise ValueError('receiver diagnostic values must be integers')
        self.code, self.details = code, details
        super().__init__(code)


def prepare_write(connector, step, state, targets, seq):
    """Stage every layer and destination index before any cache write."""
    import torch
    try:
        from glm_cache_commit import validate_destinations
    except ImportError:
        from drift.serving.glm_cache_commit import validate_destinations
    path = connector._live_in / step.name / f'{seq:06d}.npz'
    digest = publication_digest(path)
    entries = load_publication(path, {f'l{layer}': (LATENT,) for layer in targets})
    rows = next(iter(entries.values())).shape[0]
    if state['filled'] + rows > step.reserve:
        raise RuntimeError('reserved span exhausted')
    start = step.reserve_start + state['filled']
    staged = []
    for layer, part in sorted(targets.items()):
        if part.dtype != torch.uint8 or part.ndim != 3 or part.shape[-1] < PACKED:
            raise ValueError('receiver cache is not fp8_ds_mla pages')
        page, slot = connector._live_index(part, step.blocks, start, start + rows)
        validate_destinations(part, page, slot, rows)
        packed = torch.from_numpy(pack_fp8_ds_mla(entries[f'l{layer}'])).to(part.device)
        staged.append((part, page, slot, packed))
    if publication_digest(path) != digest:
        raise ValueError('publication changed during preparation')
    return rows, staged, digest


def apply_live_step(connector, step):
    """Any failure latches poison and prevents future memory writes or taps."""
    state = connector._live_worker.setdefault(step.name, {'filled': 0, 'tapped': step.reserve_start + step.reserve, 'taps': 0})
    if state.get('poisoned'):
        raise RuntimeError('live session is poisoned; use a new session')
    try:
        _apply(connector, step, state)
    except Exception:
        state['poisoned'] = True
        raise


def _apply(connector, step, state):
    import torch
    try:
        from glm_cache_commit import commit_cache
    except ImportError:
        from drift.serving.glm_cache_commit import commit_cache
    targets = connector._live_targets()
    on_gpu = any(part.is_cuda for part in targets.values())
    for seq in step.apply:
        if seq != state.get('next_sequence', 0):
            raise ValueError('live rank publication sequence mismatch')
        rows, staged, digest = prepare_write(connector, step, state, targets, seq)
        commit_cache(staged, PACKED)
        state['filled'] += rows
        acknowledge(connector, step.name, seq, rows, digest)
        state['next_sequence'] = seq + 1
    capture_taps(connector, step, state, targets, on_gpu)


def failure_message(error):
    """Export only fixed failure text, never arbitrary exception payloads."""
    detail = 'reserved span exhausted' if str(error) == 'reserved span exhausted' else 'live receiver failed; session poisoned'
    if isinstance(error, LiveReceiverError):
        detail = f'{error.code}: {detail}'
    return f'{type(error).__name__}: {detail}'


def failure_diagnostics(error):
    """Retain numeric context and stack locations without values, locals or source."""
    report = {'failure': failure_message(error), 'stack': []}
    if isinstance(error, LiveReceiverError):
        report.update(code=error.code, details=error.details)
    for frame, line in traceback.walk_tb(error.__traceback__):
        report['stack'].append({'file': os.path.basename(frame.f_code.co_filename),
                                'function': frame.f_code.co_name, 'line': line})
    return report
