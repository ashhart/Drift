"""Load only the explicitly pinned forward-v4 or reverse-v3 bridge recipe."""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from drift.serving.bridge_files import GLM_LAYERS, QWEN_LAYERS, canonical, frozen, integer, pin, require
from drift.serving.live_recipe import load_recipe
from drift.serving.worker_activation_files import private_root, temporary
from drift.translate.stacked import StackedReader


@dataclass(frozen=True)
class PinnedRecipe:
    direction: str
    reader: object
    sha256: str


def load_bridge_recipe(direction, manifest, expected_sha256, *, artifact_root, scratch, max_bytes):
    require(direction in ('forward','reverse'));pin(expected_sha256)
    root=Path(artifact_root);require(root.is_dir() and root.is_absolute() and root.resolve()==root)
    scratch=private_root(scratch);integer(max_bytes,1024*1048576)
    normalized=temporary(scratch)
    try:
        with ExitStack() as stack:
            fixed=stack.enter_context(frozen(manifest,expected_sha256,scratch,1048576))
            spec=json.loads(fixed.read_bytes())
            keys=('forward_base','forward_fanout','correction') if direction=='forward' else ('reverse_v3',)
            gain=spec['forward_gain_power' if direction=='forward' else 'reverse_gain_power']
            require(type(gain) in (int,float) and gain==(1.5 if direction=='forward' else 1.0))
            require(direction=='forward' or type(spec['reverse_copies']) is int and spec['reverse_copies']==12)
            paths={};total=0
            for key in keys:
                value=spec['artifacts'][key];require(type(value) is str)
                source=canonical(root/value if not Path(value).is_absolute() else value)
                require(total+source.stat().st_size<=max_bytes)
                paths[key]=str(stack.enter_context(frozen(source,pin(spec['sha256'][key]),scratch,max_bytes-total)))
                total+=Path(paths[key]).stat().st_size;require(total<=max_bytes)
            if direction=='forward':
                normalized.write_text(json.dumps({'artifacts':paths,'sha256':{key:spec['sha256'][key] for key in keys},'forward_gain_power':1.5}))
                normalized_hash=hashlib.sha256(normalized.read_bytes()).hexdigest()
                reader=load_recipe(normalized,'v4',GLM_LAYERS,QWEN_LAYERS,2,256)
                require(hashlib.sha256(normalized.read_bytes()).hexdigest()==normalized_hash)
                require(reader.base.input_mean.shape==(5632,))
                require(reader.metadata['sha256']=={key:spec['sha256'][key] for key in keys} and reader.gain_power==1.5)
            else:
                reader=StackedReader.load(paths['reverse_v3'],QWEN_LAYERS,GLM_LAYERS,0,512)
                require(reader.sha256==spec['sha256']['reverse_v3'])
                require(reader.input_mean.shape==(12288,) and np.isfinite(reader.input_mean).all())
                require(all(np.isfinite(value).all() for value in reader.biases.values()))
        return PinnedRecipe(direction,reader,expected_sha256)
    finally:
        normalized.unlink(missing_ok=True)
