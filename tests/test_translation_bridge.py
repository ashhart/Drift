import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from drift.serving.translation_bridge import translate_tap, translate_prefix
from drift.serving.bridge_recipe import PinnedRecipe, load_bridge_recipe
from drift.serving.bridge_files import GLM_LAYOUT, QWEN_LAYOUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folders(tmp_path):
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir(mode=0o700); output.mkdir(mode=0o700)
    return source, output


class Reverse:
    def read(self, arrays, gain):
        assert gain == 1 and all(value.shape == (1,1024) for value in arrays.values())
        assert all(np.all(value == 3) for value in arrays.values())
        return {int(key[1:]): np.ones((1,512),np.float32) for key in GLM_LAYOUT}


def tap_input(source):
    path = source/'tap.npz'
    np.savez(path, **{key:np.repeat(np.arange(1,4,dtype=np.float16)[:,None,None],512,axis=2).reshape(3,2,256) for key in QWEN_LAYOUT})
    return dict(v=1,session='qwen-turn',seq=2,op='tapped',source_worker='qwen',target_worker='glm',rows=3,next_first=13,sha256=digest(path))


def reverse(source, output, receipt, **limits):
    return translate_tap(PinnedRecipe('reverse',Reverse(),'a'*64), source/'tap.npz',receipt,
        session='qwen-turn',source_worker='qwen',target_worker='glm',seq=2,first=10,
        output=output/'reverse.npz',max_source_rows=4,max_source_bytes=1048576,
        max_output_rows=12,max_output_bytes=1048576,**limits)


def test_reverse_selects_exact_last_source_row_and_twelve_copies(tmp_path):
    source, output = folders(tmp_path); receipt = tap_input(source)
    result = reverse(source,output,receipt)
    assert (result['source_start'],result['source_stop'],result['source_rows'],result['rows']) == (12,13,1,12)
    assert result['subset'] == 'last_one_own_row' and result['copies'] == 12
    assert result['full_completion'] is False and result['source_sha256'] == receipt['sha256']
    assert result['sha256'] == digest(output/'reverse.npz')
    assert (output/'reverse.npz').stat().st_mode & 0o777 == 0o400
    with np.load(output/'reverse.npz') as arrays:
        assert set(arrays.files) == set(GLM_LAYOUT)
        assert all(arrays[key].shape == (12,512) for key in arrays.files)
    with pytest.raises((ValueError,FileExistsError)): reverse(source,output,receipt)


@pytest.mark.parametrize('fault',['source','target','session','seq','hash','rows','cursor','extra','nonfinite','layer'])
def test_tap_metadata_and_all_layers_fail_before_output(tmp_path,fault):
    source,output=folders(tmp_path); receipt=tap_input(source)
    changes={'source':{'source_worker':'forged'},'target':{'target_worker':'qwen'},'session':{'session':'other'},'seq':{'seq':True},'hash':{'sha256':'0'*64},'rows':{'rows':2},'cursor':{'next_first':14},'extra':{'text':'forged'}}
    receipt.update(changes.get(fault,{}))
    if fault in ('nonfinite','layer'):
        arrays={key:np.ones((3,2,256),np.float16) for key in QWEN_LAYOUT}
        if fault=='nonfinite': arrays['k3'][0,0,0]=np.nan
        else: arrays.pop('v3')
        np.savez(source/'tap.npz',**arrays);receipt['sha256']=digest(source/'tap.npz')
    with pytest.raises(ValueError): reverse(source,output,receipt)
    assert not (output/'reverse.npz').exists()


def prefix_input(source):
    path=source/'own-000000.npz';np.savez(path,**{key:np.ones((2,512),np.float16) for key in GLM_LAYOUT})
    publication=dict(file=path.name,sha256=digest(path),bytes=path.stat().st_size,source_seq=1,source_start=20,source_stop=22,rows=2)
    report=dict(session='glm-turn',source_worker='glm',target_worker='qwen',recipe_sha256='b'*64,evidence='EXPORTED_PREFIX',full_completion=False,final_tail='UNKNOWN',scope='newly_generated_only',new_own_inputs='NOT_EXPORTED',translation='NOT_PERFORMED',prompt_tokens=20,verified_stop=22,selected_rows=2,selected_bytes=path.stat().st_size,raw_bytes=80000,native_finished_marker=True,writes_scheduled=1,raw_publications=[],publications=[publication])
    manifest=source/'manifest.json';manifest.write_text(json.dumps(report));return manifest,report


def forward_reader(monkeypatch):
    from drift.translate.fanout import FanoutReader
    base=SimpleNamespace(writer_layers=tuple(int(k[1:]) for k in GLM_LAYOUT),input_mean=np.zeros(5632,np.float32))
    reader=object.__new__(FanoutReader)
    for key,value in dict(base=base,basis=np.zeros((5632,1),np.float32),count_w=np.zeros((1,2),np.float32),count_b=np.array([0,2],np.float32),margin=0,residual={1:None}).items(): object.__setattr__(reader,key,value)
    def read(self, arrays, gain):
        assert gain==1.5
        return {int(key[1:]):(np.ones((4,2,256),np.float32),np.ones((4,2,256),np.float32)) for key in QWEN_LAYOUT if key.startswith('k')}
    monkeypatch.setattr(FanoutReader,'read',read)
    return reader


def forward(source,output,manifest,reader,*,remaining=4,max_rows=4):
    return translate_prefix(PinnedRecipe('forward',reader,'b'*64),manifest,digest(manifest),publication_seq=0,
        session='glm-turn',source_worker='glm',target_worker='qwen',output=output/'forward.npz',
        max_source_rows=4,max_source_bytes=1048576,max_output_rows=max_rows,max_output_bytes=1048576,remaining=remaining)


def test_prefix_preserves_scope_and_separates_source_from_emitted_rows(tmp_path,monkeypatch):
    source,output=folders(tmp_path);manifest,report=prefix_input(source)
    result=forward(source,output,manifest,forward_reader(monkeypatch))
    assert (result['source_start'],result['source_stop'],result['source_rows'],result['rows'])==(20,22,2,4)
    assert result['scope']=='newly_generated_only' and result['new_own_inputs']=='NOT_EXPORTED'
    assert result['final_tail']=='UNKNOWN' and result['full_completion'] is False
    assert result['source_manifest_sha256']==digest(manifest)


@pytest.mark.parametrize('fault',['route','session','recipe','complete','tail','scope','prompt','rows','hash','bytes'])
def test_prefix_forged_provenance_or_completion_rejected(tmp_path,monkeypatch,fault):
    source,output=folders(tmp_path);manifest,report=prefix_input(source)
    changes={'route':{'source_worker':'qwen'},'session':{'session':'other'},'recipe':{'recipe_sha256':'c'*64},'complete':{'full_completion':True},'tail':{'final_tail':'COMPLETE'},'scope':{'new_own_inputs':'EXPORTED'},'prompt':{'prompt_tokens':21}}
    report.update(changes.get(fault,{}))
    if fault=='rows':report['publications'][0]['rows']=1
    if fault=='hash':report['publications'][0]['sha256']='0'*64
    if fault=='bytes':report['publications'][0]['bytes']=1
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError):forward(source,output,manifest,forward_reader(monkeypatch))
    assert not (output/'forward.npz').exists()


@pytest.mark.parametrize('remaining,max_rows',[(3,4),(4,3)])
def test_fanout_budget_refused_before_translation(tmp_path,monkeypatch,remaining,max_rows):
    source,output=folders(tmp_path);manifest,_=prefix_input(source);reader=forward_reader(monkeypatch)
    def forbidden(*args):raise AssertionError('translator dispatched')
    monkeypatch.setattr(type(reader),'read',forbidden)
    with pytest.raises(ValueError):forward(source,output,manifest,reader,remaining=remaining,max_rows=max_rows)
    assert not (output/'forward.npz').exists()


def test_source_is_immutable_before_arrays_are_read(tmp_path,monkeypatch):
    from drift.serving import bridge_sources
    source,output=folders(tmp_path);receipt=tap_input(source);original=bridge_sources.load_publication
    def mutate(path,*args,**kwargs):
        (source/'tap.npz').write_bytes(b'tampered')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(bridge_sources,'load_publication',mutate)
    with pytest.raises(ValueError):reverse(source,output,receipt)
    assert not (output/'reverse.npz').exists()


@pytest.mark.parametrize('fault',['nonfinite','extra','shape'])
def test_translated_layers_fail_before_immutable_publication(tmp_path,monkeypatch,fault):
    source,output=folders(tmp_path);receipt=tap_input(source)
    def broken(self,arrays,gain):
        result={int(key[1:]):np.ones((1,512),np.float32) for key in GLM_LAYOUT}
        if fault=='nonfinite':result[3][0,0]=np.inf
        if fault=='extra':result[99]=np.ones((1,512),np.float32)
        if fault=='shape':result[3]=np.ones((2,512),np.float32)
        return result
    monkeypatch.setattr(Reverse,'read',broken)
    with pytest.raises(ValueError):reverse(source,output,receipt)
    assert not (output/'reverse.npz').exists()


def test_byte_limit_precedes_reverse_reader_dispatch(tmp_path,monkeypatch):
    source,output=folders(tmp_path);receipt=tap_input(source)
    def forbidden(*args):raise AssertionError('translator dispatched')
    monkeypatch.setattr(Reverse,'read',forbidden)
    with pytest.raises(ValueError):
        translate_tap(PinnedRecipe('reverse',Reverse(),'a'*64),source/'tap.npz',receipt,
            session='qwen-turn',source_worker='qwen',target_worker='glm',seq=2,first=10,
            output=output/'reverse.npz',max_source_rows=4,max_source_bytes=1048576,max_output_rows=12,max_output_bytes=1)
    assert not (output/'reverse.npz').exists()


def test_actual_turn_outbox_manifest_connects_to_forward_bridge(tmp_path,monkeypatch):
    import time
    from drift.serving.glm_turn_outbox import TurnOutbox
    source,output=folders(tmp_path);native=tmp_path/'native';native.mkdir(mode=0o700)
    folder=native/'glm-turn';folder.mkdir(mode=0o700)
    collector=TurnOutbox(dict(memory_root=str(source),source_worker='glm',target_worker='qwen',recipe_sha256='b'*64,max_raw_rows=32,max_raw_bytes=1048576,max_rows=4,max_bytes=1048576,max_publications=4),GLM_LAYOUT,native)
    collector.begin('glm-turn',dict(verified=True,prompt_tokens=20,reserve_start=3,reserve_tokens=2),4)
    np.savez(folder/'000000.npz',start=np.array(5),stop=np.array(22),**{key:np.ones((17,512),np.float16) for key in GLM_LAYOUT})
    (folder/'finished').write_text(json.dumps(dict(failed='',writes_scheduled=1)))
    report=collector.finish('glm-turn',4,time.monotonic()+2)
    result=forward(source,output,source/'glm-turn'/'manifest.json',forward_reader(monkeypatch))
    assert result['source_manifest_sha256']==report['manifest_sha256'] and result['rows']==4


def test_forward_byte_limit_precedes_translation(tmp_path,monkeypatch):
    source,output=folders(tmp_path);manifest,_=prefix_input(source);reader=forward_reader(monkeypatch)
    def forbidden(*args):raise AssertionError('translator dispatched')
    monkeypatch.setattr(type(reader),'read',forbidden)
    with pytest.raises(ValueError):
        translate_prefix(PinnedRecipe('forward',reader,'b'*64),manifest,digest(manifest),publication_seq=0,
            session='glm-turn',source_worker='glm',target_worker='qwen',output=output/'forward.npz',
            max_source_rows=4,max_source_bytes=1048576,max_output_rows=4,max_output_bytes=1,remaining=4)
