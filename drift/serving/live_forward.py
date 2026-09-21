"""Budget emitted reader rows separately from the writer's source cursor."""
import math

import numpy as np

from drift.serving.live_publication import MAX_BYTES, count, wire_arrays
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader


def emitted_rows(reader, latents, source_rows):
    translator = getattr(reader, 'translator', reader)
    fan = translator.fan if isinstance(translator, CorrectedFanoutReader) else translator
    if not isinstance(fan, FanoutReader):
        return source_rows
    base = fan.base
    if set(latents) != set(base.writer_layers):
        raise ValueError('forward source layer set mismatch')
    if any(value.ndim != 2 or len(value) != source_rows or not np.isfinite(value).all() for value in latents.values()):
        raise ValueError('forward source row layout mismatch')
    stacked = np.concatenate([latents[layer] for layer in base.writer_layers], axis=1)
    if stacked.shape[1:] != base.input_mean.shape:
        raise ValueError('forward source width mismatch')
    spans = fan.spans(stacked - base.input_mean)
    if spans.shape != (source_rows,) or spans.dtype.kind not in 'iu' or (spans < 1).any():
        raise ValueError('invalid fan-out span prediction')
    return sum(int(value) for value in spans)


def forward_publication(reader, latents, source_rows, layouts, gain, *, max_rows, remaining):
    source_rows = count(source_rows, positive=True)
    rows = emitted_rows(reader, latents, source_rows)
    if rows > count(max_rows, positive=True) or rows > count(remaining):
        raise ValueError('emitted forward rows exceed publication or receiver reserve limit')
    if rows * sum(math.prod(shape) for shape in layouts.values()) * 2 > MAX_BYTES:
        raise ValueError('emitted forward rows exceed wire byte limit')
    entries = reader.read(latents, gain)
    payload = {f'{kind}{layer}': pair[index] for layer, pair in entries.items() for index, kind in enumerate(('k', 'v'))}
    return wire_arrays(payload, layouts, rows), rows
