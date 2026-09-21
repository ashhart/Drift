"""Load an explicitly selected, hash-pinned forward translation recipe."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.stacked import StackedReader


def file_sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


@dataclass(frozen=True)
class LiveRecipe:
    translator: object
    base: StackedReader
    gain_power: float
    metadata: dict

    @property
    def kv_heads(self):
        return self.base.kv_heads

    @property
    def head_dim(self):
        return self.base.head_dim

    def read(self, latents, gain_power):
        if gain_power != self.gain_power:
            raise ValueError('gain differs from the pinned forward recipe')
        return self.translator.read(latents, gain_power)


def validate_fanout(fan):
    base, rank = fan.base, fan.basis.shape[1] if fan.basis.ndim == 2 else 0
    width = 2 * base.kv_heads * base.head_dim * len(base.reader_layers)
    if (fan.basis.shape != (len(base.input_mean), rank) or rank <= 0 or fan.count_w.ndim != 2
            or fan.count_w.shape[0] != rank or fan.count_w.shape[1] <= 0
            or fan.count_b.shape != (fan.count_w.shape[1],) or not math.isfinite(fan.margin)):
        raise ValueError('fan-out classifier layout mismatch')
    if (sorted(fan.residual) != list(range(1, len(fan.residual) + 1))
            or len(fan.residual) >= fan.count_w.shape[1]):
        raise ValueError('fan-out residual sequence is incomplete')
    if any(value.shape != (rank, width) for value in fan.residual.values()):
        raise ValueError('fan-out residual layout mismatch')
    if not all(np.isfinite(value).all() for value in (fan.basis, fan.count_w, fan.count_b, *fan.residual.values())):
        raise ValueError('nonfinite fan-out parameters')


def load_recipe(manifest, kind, writer_layers, reader_layers, kv_heads, head_dim):
    if kind not in ('base', 'fanout', 'v4'):
        raise ValueError('unsupported live forward recipe')
    manifest = Path(manifest)
    manifest_bytes = manifest.read_bytes()
    spec = json.loads(manifest_bytes)
    gain = spec['forward_gain_power']
    if type(gain) not in (int, float) or not math.isfinite(gain):
        raise ValueError('forward gain must be a finite number')
    keys = ['forward_base'] + (['forward_fanout'] if kind != 'base' else []) + (['correction'] if kind == 'v4' else [])
    paths, pins = {}, {}
    for key in keys:
        paths[key], pins[key] = Path(spec['artifacts'][key]), spec['sha256'][key]
        if not isinstance(pins[key], str) or not re.fullmatch('[0-9a-f]{64}', pins[key]):
            raise ValueError('forward artifact requires a SHA-256 pin')
        if file_sha256(paths[key]) != pins[key]:
            raise ValueError('forward artifact differs from its SHA-256 pin')
    base = StackedReader.load(paths['forward_base'], writer_layers, reader_layers, kv_heads, head_dim)
    if base.input_mean.ndim != 1 or not all(np.isfinite(value).all() for value in (base.input_mean, *base.biases.values())):
        raise ValueError('nonfinite base translator parameters')
    translator, loaded = base, {'forward_base': base.sha256}
    if kind != 'base':
        translator = FanoutReader.load(paths['forward_fanout'], base)
        validate_fanout(translator)
        loaded['forward_fanout'] = translator.sha256
    if kind == 'v4':
        translator = CorrectedFanoutReader.load(paths['correction'], translator)
        loaded['correction'] = translator.sha256
    if loaded != pins or any(file_sha256(paths[key]) != pins[key] for key in keys):
        raise ValueError('forward artifacts changed while loading')
    metadata = {'kind': kind, 'sha256': pins, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'gain_power': gain, 'live_qualification': 'BLOCKED'}
    return LiveRecipe(translator, base, gain, metadata)
