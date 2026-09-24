import hashlib
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import time
import numpy as np
import pytest


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory(prefix='gown-', dir='/tmp') as value:
        path=Path(value).resolve(); path.chmod(0o700)
        yield path


def write_snapshot(root, version):
    path=root/f'v{version}.npz'; np.savez(path,l3=np.ones((2,512),dtype=np.float32)*version)
    return dict(v=1,session='owned',version=version,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=2,source_worker='qwen',target_worker='glm',recipe_sha256=hashlib.sha256(b'{}').hexdigest())


def configuration(root):
    initial=write_snapshot(root,1); (root/'recipe.json').write_bytes(b'{}')
    return dict(worker='glm',pins={'model_id':'fixture','model_sha256':'a'*64,'translator_sha256':'b'*64},
                memory_mode='linked_snapshot',experimental_multi_turn=True,
                limits=dict(max_input_bytes=65536,max_output_tokens=8,max_session_tokens=128,max_turns=2,deadline_ms=4000),
                restoration=dict(snapshot_path=str(root/initial['path']),snapshot_sha256=initial['sha256'],rows=2,layers=[3],
                                 max_rows=4,source_worker='qwen',target_worker='glm',route_path=str(root/'route'),route_sha256='c'*64),
                owner_bank=dict(root=str(root),session='owned',recipe_path=str(root/'recipe.json'),recipe_sha256=initial['recipe_sha256'],
                                max_rows=4,max_bytes=16384,max_versions=2,max_total_bytes=32768,initial=initial),
                owner_control=dict(root=str(root),socket_name='owner.sock',evidence_name='bank.json'))


def start(root):
    config=configuration(root); path=root/'config.json'; path.write_text(json.dumps(config))
    runner=root/'fixture.py'
    runner.write_text('''import sys,json,time
from pathlib import Path
from drift.serving.glm_owner_worker import serve_owner
from drift.serving.glm_session import GlmSession
from drift.serving.glm_restore_turn import TurnRestorer,RestoringTransport
import drift.serving.glm_snapshot_binding as binding
root=Path(sys.argv[1]).parent
class Delivery:
 def health(self,**kw):pass
 def admit(self,*a,**kw):pass
 def wait_applied(self,seq,rows,digest,**kw):return [{'rank':r,'world_size':2,'sequence':seq,'rows':rows,'sha256':digest} for r in (0,1)]
class Publication:
 def __init__(self,*args):self.delivery=Delivery()
 def stage(self,**kw):pass
 def publish(self,**kw):pass
class Route:
 spec={'peer':'fixture'}
 def validate(self):pass
 def bind(self,x):return x
class Prefix:
 def verify(self,a,b,rows,timeout):return {'verified':True,'own_tokens':10,'prompt_tokens':10+rows,'reserve_start':3,'reserve_tokens':rows}
 def close(self):pass
class Http:
 def __init__(self):self.turn=0;self.closed=False
 def count_tokens(self,body,timeout):return 12
 def stream(self,body,timeout):
  self.turn+=1
  if self.turn==1:
   (root/'entered').touch();deadline=time.monotonic()+timeout
   while not (root/'release').exists() and not self.closed and time.monotonic()<deadline:time.sleep(.01)
   if self.closed:raise ValueError('closed')
   delta={'tool_calls':[{'index':0,'id':'c1','type':'function','function':{'name':'echo','arguments':'{}'}}]};reason='tool_calls'
  else:delta={'content':'synthetic'};reason='stop'
  yield {'choices':[{'delta':delta,'finish_reason':None}]}
  yield {'choices':[{'delta':{},'finish_reason':reason}]}
  yield {'choices':[],'usage':{'prompt_tokens':12,'completion_tokens':2}}
 def close(self):self.closed=True
 def cancel(self):self.close()
def factory(config):
 session=GlmSession(Http(),model='fixture',recovery=lambda:None)
 session.backend='fixture';session.capabilities['native_state']='fixture'
 fixed=TurnRestorer(Prefix(),None,rows=2,digest=config['restoration']['snapshot_sha256'],root=root,deadline=lambda:time.monotonic()+4,max_input_bytes=65536)
 session.transport=RestoringTransport(session.transport,fixed);session.restoration=fixed
 session.restoration_ownership={'source_worker':'qwen','target_worker':'glm'}
 return session
binding.PublicationTransport=Publication
code=serve_owner(json.load(open(sys.argv[1])),backend_factory=factory,route_factory=lambda *a:Route())
raise SystemExit(code)
''')
    args=[sys.executable,'-m','drift.serving.worker_stdio_launcher','--evidence',str(root/'termination.json'),
          '--wall-seconds','5','--stop-timeout','.1','--',sys.executable,str(runner),str(path)]
    process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])})
    return process,config


def send(process,seq,op,payload):
    process.stdin.write(json.dumps(dict(v=1,session='owned',worker='glm',seq=seq,op=op,payload=payload)).encode()+b'\n');process.stdin.flush()


def read(process):
    assert select.select([process.stdout],[],[],3)[0]
    line=process.stdout.readline();assert line,process.stderr.read().decode()
    return json.loads(line)


def opened(process,config):
    send(process,1,'open',{**config['pins'],'limits':config['limits'],'system_prompt':['own'],
                         'tools':[{'name':'echo','description':'fixed','parameters':{'type':'object'}}]})
    assert read(process)['op']=='opened'
    send(process,2,'own_prompt',{'text':'synthetic'});assert read(process)['op']=='own_prompt_ack'
    peer=socket.socket(socket.AF_UNIX);peer.settimeout(2);peer.connect(str(Path(config['owner_control']['root'])/'owner.sock'))
    return peer


def finish(process,root):
    process.wait(timeout=5)
    receipt=json.loads((root/'termination.json').read_text())
    assert receipt['child_reaped'] and not receipt['process_group_alive']
    return json.loads((root/'bank.json').read_text())


def test_midturn_publication_is_held_for_next_native_tool_turn(root):
    process,config=start(root)
    try:
        with opened(process,config) as peer:
            send(process,3,'stream',{'max_tokens':8})
            deadline=time.monotonic()+2
            while not (root/'entered').exists() and time.monotonic()<deadline:time.sleep(.01)
            assert (root/'entered').exists()
            def applied(version, digest):
                peer.sendall(json.dumps({'op':'applied','version':version,'sha256':digest}).encode()+b'\n')
                return json.loads(peer.recv(4096))
            assert applied(1,config['restoration']['snapshot_sha256'])['state']=='PENDING'
            peer.sendall(json.dumps({'op':'publish','snapshot':write_snapshot(root,2)}).encode()+b'\n')
            assert json.loads(peer.recv(4096))['version']==2
            (root/'release').touch()
            assert read(process)['op']=='tool_call';assert read(process)['op']=='terminal'
            receipt=applied(1,config['restoration']['snapshot_sha256'])
            assert receipt['state']=='APPLIED' and len(receipt['receipts'])==2
            assert receipt['rows']==2 and receipt['version']==1
            send(process,4,'tool_result',{'call_id':'c1','text':'echo','is_error':False});assert read(process)['op']=='tool_result_ack'
            send(process,5,'stream',{'max_tokens':8});assert read(process)['op']=='text';assert read(process)['op']=='terminal'
            send(process,6,'close',{});assert read(process)['op']=='closed'
            result=finish(process,root)
            assert result['status']=='PASSED' and [x['snapshot']['version'] for x in result['restoration']['turns']]==[1,2]
            assert not (root/'owner.sock').exists()
    finally:
        if process.poll() is None:process.kill();process.wait()


@pytest.mark.parametrize('kind',['malformed','eof','abort'])
def test_owner_failure_poisoned_and_reaped_while_own_stdin_open(root,kind):
    process,config=start(root)
    try:
        peer=opened(process,config)
        send(process,3,'stream',{'max_tokens':8})
        deadline=time.monotonic()+2
        while not (root/'entered').exists() and time.monotonic()<deadline:time.sleep(.01)
        assert (root/'entered').exists()
        if kind=='malformed':peer.sendall(b'{"op":"publish","secret":"NEVER_EXPORT"}\n')
        elif kind=='abort':peer.sendall(b'{"op":"abort"}\n')
        else:peer.close()
        result=finish(process,root)
        assert result['status']=='FAILED' and result['bank_poisoned'] and result['control_failed']
        assert result['restoration']['completed_turns']==0
        assert 'NEVER_EXPORT' not in json.dumps(result)
        peer.close()
    finally:
        if process.poll() is None:process.kill();process.wait()


@pytest.mark.parametrize('change',['recipe','initial_hash','session','target','mode'])
def test_pin_mismatch_is_rejected_before_backend_construction(root,change):
    from drift.serving.glm_owner_config import create_owner
    config=configuration(root)
    if change=='recipe':config['owner_bank']['recipe_sha256']='f'*64
    elif change=='initial_hash':config['restoration']['snapshot_sha256']='f'*64
    elif change=='session':config['owner_bank']['session']='another'
    elif change=='target':config['worker']='another'
    else:config['memory_mode']='no_link'
    calls=[]
    with pytest.raises(ValueError):create_owner(config,backend_factory=lambda value:calls.append(value))
    assert calls==[]


def test_own_protocol_session_cannot_rebind_owner_bank(root):
    process,config=start(root)
    try:
        frame=dict(v=1,session='wrong',worker='glm',seq=1,op='open',payload={**config['pins'],'limits':config['limits'],
                   'system_prompt':['own'],'tools':[]})
        process.stdin.write(json.dumps(frame).encode()+b'\n');process.stdin.flush()
        assert read(process)['op']=='error'
        result=finish(process,root)
        assert result['status']=='FAILED' and result['bank_poisoned']
    finally:
        if process.poll() is None:process.kill();process.wait()
