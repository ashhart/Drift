import hashlib,json,shutil,sys,time
from pathlib import Path
import pytest
from test_linked_round import actor
from test_translation_bridge import Reverse,tap_input,prefix_input,forward_reader
from drift.serving.bridge_recipe import PinnedRecipe
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/omp'))
from linked_round_process import rendezvous
from linked_round_exchange import Exchange


def setup(tmp_path,monkeypatch,empty=False):
    from linked_round_remote import LOCATE
    import linked_round_exchange as module
    q=tmp_path/'q';q.mkdir(mode=0o700);g=tmp_path/'g';g.mkdir(mode=0o700)
    tap=tap_input(q);tap.update(session='qwen',seq=1,next_first=3)
    manifest,report=prefix_input(g);report['session']='restore-test'
    if empty:report['selected_rows']=0;report['publications']=[]
    manifest.write_text(json.dumps(report));files={'/q/round-own.npz':q/'tap.npz','/g/outbox/restore-test/manifest.json':manifest,'/g/outbox/restore-test/own-000000.npz':g/'own-000000.npz'}
    events=[]
    class Remote:
        def __init__(self,command,deadline):self.name=command['name'];self.deadline=deadline
        def download(self,remote,local,maximum,expected=None):
            raw=files[remote].read_bytes();assert len(raw)<=maximum
            digest=hashlib.sha256(raw).hexdigest()
            if expected is not None:assert digest==expected
            Path(local).write_bytes(raw);return digest
        def upload(self,local,remote,maximum,expected,scratch):
            raw=Path(local).read_bytes();assert len(raw)<=maximum and hashlib.sha256(raw).hexdigest()==expected
            target=tmp_path/('upload-'+self.name+'.npz');target.write_bytes(raw);files[remote]=target;Path(scratch).write_bytes(b'')
        def call(self,program,args,target,maximum):
            assert program==LOCATE
            Path(target).write_text(json.dumps({'directory':'restore-test','sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    class Forward:
        def __init__(self,remote,path,local):self.name=remote.name
        def request(self,frame):
            events.append((self.name,frame['op']))
            if frame['op']=='tap':return dict(tap)
            if frame['op']=='publish':
                value=frame['snapshot'];return {'op':'published',**{key:v for key,v in value.items() if key not in ('v','path')},'bytes':files['/g/round-v2.npz'].stat().st_size}
            return dict(v=1,session='qwen',seq=2,source_worker='glm',target_worker='qwen',op='appended',rows=frame['rows'],sha256=frame['sha256'],foreign_total=frame['rows'])
        def close(self):return {'reaped':True}
    monkeypatch.setattr(module,'Remote',Remote);monkeypatch.setattr(module,'Forward',Forward)
    profile={'socket_root':str(tmp_path),'endpoints':{'glm':dict(path='/g.sock',memory_root='/g',outbox_root='/g/outbox',worker='glm',session='glm'),'qwen':dict(path='/q.sock',memory_root='/q',worker='qwen',session='qwen')}}
    entries={name:{'command':{'name':name}} for name in ('glm','qwen')};transfer=tmp_path/'transfer';transfer.mkdir(mode=0o700)
    exchange=Exchange(profile,entries,{'reverse':PinnedRecipe('reverse',Reverse(),'a'*64),'forward':PinnedRecipe('forward',forward_reader(monkeypatch),'b'*64)},transfer,time.monotonic()+3)
    return exchange,events


def test_two_children_exchange_actual_arrays_through_real_bridge_then_resume(tmp_path,monkeypatch):
    exchange,events=setup(tmp_path,monkeypatch);one,r1=actor(tmp_path,'one');two,r2=actor(tmp_path,'two')
    try:
        result=rendezvous({'glm':(one,r1),'qwen':(two,r2)},exchange,time.monotonic()+2)
        assert events==[('qwen','tap'),('glm','publish'),('qwen','append')]
        assert exchange.receipt['reverse']['rows']==12 and exchange.receipt['forward']['rows']==4
        assert exchange.receipt['forward']['final_tail']=='UNKNOWN'
        assert all(v['exit_code']==0 for v in result.values())
    finally:exchange.close();one.close(time.monotonic()+1);two.close(time.monotonic()+1)


def test_empty_native_prefix_blocks_both_release_files_without_extra_generation(tmp_path,monkeypatch):
    exchange,events=setup(tmp_path,monkeypatch,empty=True);one,r1=actor(tmp_path,'one');two,r2=actor(tmp_path,'two')
    try:
        with pytest.raises(ValueError):rendezvous({'glm':(one,r1),'qwen':(two,r2)},exchange,time.monotonic()+2)
        assert events==[('qwen','tap'),('glm','publish')]
        assert not (r1/'release.json').exists() and not (r2/'release.json').exists()
    finally:exchange.close();one.close(time.monotonic()+1);two.close(time.monotonic()+1)


def test_close_attempts_every_forwarder_when_one_fails():
    from linked_round_exchange import Exchange
    calls=[]
    class Broken:
        def close(self):calls.append('broken');raise OSError('private detail')
    class Healthy:
        def close(self):calls.append('healthy');return {'reaped':True}
    exchange=Exchange.__new__(Exchange);exchange.forwards={'glm':Broken(),'qwen':Healthy()}
    result=exchange.close()
    assert calls==['broken','healthy']
    assert result=={'glm':{'reaped':False,'error':'FORWARD_CLEANUP_FAILED'},'qwen':{'reaped':True}}
