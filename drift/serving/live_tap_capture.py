"""Capture verified taps and retain one private candidate tail before cache reuse."""
import os

import numpy as np

try:
    from glm53_handoff import dequantize_fp8_ds_mla
except ImportError:
    from drift.serving.glm53_handoff import dequantize_fp8_ds_mla


def save_arrays(path, **arrays):
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def capture_range(connector, step, targets, start, stop):
    if not 0 < stop - start <= 4096:
        raise ValueError('live tap range outside bound')
    arrays = {}
    for layer, part in sorted(targets.items()):
        page, slot = connector._live_index(part, step.blocks, start, stop)
        values = dequantize_fp8_ds_mla(part[page, slot].cpu().numpy())
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float16).max).any():
            raise ValueError('outbound live entries cannot be represented as finite float16')
        arrays[f'l{layer}'] = values.astype(np.float16)
    return arrays


def capture_taps(connector, step, state, targets, on_gpu):
    if not step.tap or connector._tp()[0] != 0:
        return
    if step.before < 0 or step.after < step.before:
        raise ValueError('live tap frontier is invalid')
    if step.before < state['tapped'] and state['taps']:
        raise ValueError('live tap frontier rolled back past publication')
    if step.after <= state['tapped']:
        return
    if on_gpu:
        import torch
        torch.cuda.synchronize()
    folder = connector._live_out / step.name
    folder.mkdir(parents=True, exist_ok=True)
    if step.before - state['tapped'] >= connector._live_tap_every:
        arrays = capture_range(connector, step, targets, state['tapped'], step.before)
        save_arrays(folder / f"{state['taps']:06d}.npz", start=np.array(state['tapped']), stop=np.array(step.before), **arrays)
        state['tapped'], state['taps'] = step.before, state['taps'] + 1
    arrays = capture_range(connector, step, targets, state['tapped'], step.after)
    pending = folder / '.pending'
    pending.mkdir(mode=0o700, exist_ok=True)
    save_arrays(pending / 'tail.npz', start=np.array(state['tapped']), stop=np.array(step.after),
                verified=np.array(step.before), sequence=np.array(state['taps']), **arrays)
