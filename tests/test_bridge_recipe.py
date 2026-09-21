import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from drift.serving.bridge_recipe import load_bridge_recipe


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,direction):
    root=tmp_path/'artifacts';root.mkdir();scratch=tmp_path/'private';scratch.mkdir(mode=0o700)
    keys=('forward_base','forward_fanout','correction') if direction=='forward' else ('reverse_v3',)
    paths={key:root/f'{key}.npz' for key in keys}
    for key,path in paths.items():path.write_bytes(key.encode())
    spec=dict(artifacts={key:path.name for key,path in paths.items()},sha256={key:digest(path) for key,path in paths.items()},forward_gain_power=1.5,reverse_gain_power=1.0,reverse_copies=12)
    manifest=root/'recipe.json';manifest.write_text(json.dumps(spec))
    return root,scratch,manifest,paths,spec


@pytest.mark.parametrize('direction',['forward','reverse'])
@pytest.mark.parametrize('mutation',[None,'source','snapshot','manifest','normalized'])
def test_recipe_hashes_are_checked_before_and_after_loader(tmp_path,monkeypatch,direction,mutation):
    from drift.serving import bridge_recipe
    root,scratch,manifest,paths,spec=fixture(tmp_path,direction);expected=digest(manifest)
    def mutate(snapshot):
        if mutation=='source':next(iter(paths.values())).write_bytes(b'tampered')
        if mutation=='snapshot':
            Path(snapshot).chmod(0o600);Path(snapshot).write_bytes(b'tampered')
        if mutation=='manifest':manifest.write_bytes(b'tampered')
    def forward(path,kind,*layout):
        assert kind=='v4' and layout==(tuple(range(3,44,4)),tuple(range(3,48,4)),2,256)
        data=json.loads(Path(path).read_bytes());mutate(data['artifacts']['forward_base'])
        if mutation=='normalized':Path(path).write_bytes(b'tampered')
        return SimpleNamespace(metadata={'sha256':spec['sha256']},gain_power=1.5,base=SimpleNamespace(input_mean=np.zeros(5632)))
    def reverse(path,*layout):
        assert layout==(tuple(range(3,48,4)),tuple(range(3,44,4)),0,512)
        value=SimpleNamespace(sha256=digest(Path(path)),input_mean=np.zeros(12288),biases={3:np.zeros(512)})
        mutate(path);return value
    monkeypatch.setattr(bridge_recipe,'load_recipe',forward)
    monkeypatch.setattr(bridge_recipe.StackedReader,'load',reverse)
    invoke=lambda:load_bridge_recipe(direction,manifest,expected,artifact_root=root,scratch=scratch,max_bytes=1024)
    if mutation is None or direction=='reverse' and mutation=='normalized':
        assert invoke().sha256==expected
    else:
        with pytest.raises(ValueError):invoke()
    assert not list(scratch.iterdir())


@pytest.mark.parametrize('fault',['manifest','artifact','budget','gain','copies','symlink'])
def test_recipe_rejects_bad_pins_limits_and_recipe_before_loading(tmp_path,monkeypatch,fault):
    from drift.serving import bridge_recipe
    root,scratch,manifest,paths,spec=fixture(tmp_path,'reverse');expected=digest(manifest)
    if fault=='manifest':expected='0'*64
    if fault=='artifact':next(iter(paths.values())).write_bytes(b'tampered')
    if fault=='gain':spec['reverse_gain_power']=2
    if fault=='copies':spec['reverse_copies']=True
    if fault=='symlink':
        target=root/'alias.npz';target.symlink_to(paths['reverse_v3']);spec['artifacts']['reverse_v3']=target.name
    if fault in ('gain','copies','symlink'):
        manifest.write_text(json.dumps(spec));expected=digest(manifest)
    def forbidden(*args):raise AssertionError('loader reached')
    monkeypatch.setattr(bridge_recipe.StackedReader,'load',forbidden)
    with pytest.raises(ValueError):load_bridge_recipe('reverse',manifest,expected,artifact_root=root,scratch=scratch,max_bytes=1 if fault=='budget' else 1024)
