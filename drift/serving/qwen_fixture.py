"""Convert one controlled final Qwen prompt row with the frozen reverse-v3 translator."""
import hashlib
from pathlib import Path

import numpy as np

from drift.serving.live_publication import load_publication, wire_arrays
from drift.translate.stacked import StackedReader

PUBLIC_NOTE = 'API: GET /health returns HTTP 200.'
REVERSE_SHA256 = '36c352797d936aca6468d4066c1be93fea99ab0185f7505e731874dc5e7be8b9'
QWEN_LAYERS = tuple(3 + 4*i for i in range(12))
GLM_LAYERS = tuple(3 + 4*i for i in range(11))


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): value.update(block)
    return value.hexdigest()


def build_fixture(tap_path, translator_path, output_path, tokens):
    if type(tokens) is not int or not 1 <= tokens <= 32:
        raise ValueError('source token count exceeds fixture limit')
    if sha256_file(translator_path) != REVERSE_SHA256:
        raise ValueError('translator hash differs from frozen reverse-v3')
    layouts = {f'{kind}{layer}': (2,256) for layer in QWEN_LAYERS for kind in ('k','v')}
    taps = load_publication(tap_path, layouts, max_rows=32)
    if any(len(value) != tokens for value in taps.values()):
        raise ValueError('source rows differ from controlled token count')
    final = {layer: np.concatenate([taps[f'{kind}{layer}'][-1:].reshape(1,-1) for kind in ('k','v')],axis=1)
             for layer in QWEN_LAYERS}
    reader = StackedReader.load(translator_path,QWEN_LAYERS,GLM_LAYERS,kv_heads=0,head_dim=512)
    if reader.sha256 != REVERSE_SHA256: raise ValueError('translator changed while loading')
    translated = reader.read(final,1.0)
    checked = wire_arrays({f'l{layer}': value for layer,value in translated.items()},
                          {f'l{layer}': (512,) for layer in GLM_LAYERS},1)
    with Path(output_path).open('xb') as stream:
        np.savez(stream,**{name:np.repeat(value,12,axis=0) for name,value in checked.items()})
    return {'source_text_sha256':hashlib.sha256(PUBLIC_NOTE.encode()).hexdigest(),
            'source_tap_sha256':sha256_file(tap_path),'translator_sha256':REVERSE_SHA256,
            'fixture_sha256':sha256_file(output_path),'source_tokens':tokens,'source_row':tokens-1,
            'generated_tokens':0,'source_layers':12,'source_shape_per_array':[tokens,2,256],
            'fixture_layers':11,'fixture_shape_per_array':[12,512],'copies':12,'gain_power':1.0}
