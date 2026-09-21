import hashlib,json,os,sys,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/omp'))
from linked_round_process import Actor, rendezvous
from linked_round_evidence import check_bank


def actor(tmp_path,name,*,fail=False):
    root=tmp_path/name;root.mkdir(mode=0o700)
    code="""import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);fail=sys.argv[2]=='True'
(root/'ready.json').write_text(json.dumps({'v':1,'phase':'tool_boundary'})+'\\n')
if fail:raise SystemExit(2)
end=time.monotonic()+3
while not (root/'release.json').exists():
 if time.monotonic()>end:raise SystemExit(3)
 time.sleep(.005)
value=json.loads((root/'release.json').read_text())
assert value=={'v':1,'action':'resume','ready_sha256':hashlib.sha256((root/'ready.json').read_bytes()).hexdigest()}
(root/'facts.json').write_text(json.dumps({'tool_calls':1,'other_tools':0,'assistant_turns':2,'input_tokens':5,'output_tokens':3,'diagnostics':[]}))
print('discarded generated fixture text')
"""
    return Actor([sys.executable,'-c',code,str(root),str(fail)],root,dict(os.environ)),root


def test_two_actual_children_park_exchange_once_then_release(tmp_path):
    one,r1=actor(tmp_path,'one');two,r2=actor(tmp_path,'two');seen=[]
    try:
        receipt=rendezvous({'one':(one,r1),'two':(two,r2)},lambda:seen.append('exchange'),time.monotonic()+2)
        assert seen==['exchange'] and all(x['exit_code']==0 and x['reaped'] for x in receipt.values())
        assert all(x['stdout']['bytes']>0 and 'text' not in x for x in receipt.values())
    finally:one.close(time.monotonic()+1);two.close(time.monotonic()+1)


def test_exchange_failure_never_releases_either_child_and_cleanup_reaps(tmp_path):
    one,r1=actor(tmp_path,'one');two,r2=actor(tmp_path,'two')
    def fail():raise ValueError('EMPTY_PREFIX')
    with pytest.raises(ValueError):rendezvous({'one':(one,r1),'two':(two,r2)},fail,time.monotonic()+2)
    for child,root in ((one,r1),(two,r2)):
        child.close(time.monotonic()+1)
        assert child.process.poll() is not None and not (root/'release.json').exists()


def test_dead_actor_prevents_exchange_and_peer_release(tmp_path):
    one,r1=actor(tmp_path,'one',fail=True);two,r2=actor(tmp_path,'two');seen=[]
    time.sleep(.06)
    with pytest.raises(ValueError):rendezvous({'one':(one,r1),'two':(two,r2)},lambda:seen.append(1),time.monotonic()+1)
    one.close(time.monotonic()+1);two.close(time.monotonic()+1)
    assert not seen and not (r2/'release.json').exists()


def bank():
    turns=[]
    for version,digest in ((1,'a'*64),(2,'b'*64)):
        turns.append(dict(session=f'restore-{version}',snapshot=dict(version=version,sha256=digest,rows=12),snapshot_sha256=digest,linked=True,receipts=[dict(rank=rank,world_size=2,sequence=0,rows=12,sha256=digest) for rank in (0,1)]))
    return dict(status='PASSED',control_failed=False,owner_thread_joined=True,restoration=dict(completed_turns=2,turns=turns,failed=True))


@pytest.mark.parametrize('fault',[None,'version','rank','hash','same_session','missing'])
def test_final_bank_requires_both_versions_and_both_ranks(fault):
    value=bank()
    if fault=='version':value['restoration']['turns'][1]['snapshot']['version']=1
    if fault=='rank':value['restoration']['turns'][1]['receipts'][1]['rank']=0
    if fault=='hash':value['restoration']['turns'][1]['snapshot_sha256']='c'*64
    if fault=='same_session':value['restoration']['turns'][1]['session']='restore-1'
    if fault=='missing':value['restoration']['turns'][1]['receipts'].pop()
    if fault:
        with pytest.raises(ValueError):check_bank(value,'a'*64,'b'*64)
    else:assert check_bank(value,'a'*64,'b'*64)['versions']==[1,2]


def test_expired_round_never_releases_waiting_children(tmp_path):
    one,r1=actor(tmp_path,'one');two,r2=actor(tmp_path,'two')
    with pytest.raises(TimeoutError):rendezvous({'one':(one,r1),'two':(two,r2)},lambda:None,time.monotonic()-1)
    one.close(time.monotonic()+1);two.close(time.monotonic()+1)
    assert not (r1/'release.json').exists() and not (r2/'release.json').exists()


def test_exchange_that_runs_over_deadline_cannot_release(tmp_path):
    one,r1=actor(tmp_path,'one');two,r2=actor(tmp_path,'two')
    with pytest.raises(TimeoutError):rendezvous({'one':(one,r1),'two':(two,r2)},lambda:time.sleep(.1),time.monotonic()+.07)
    one.close(time.monotonic()+1);two.close(time.monotonic()+1)
    assert not (r1/'release.json').exists() and not (r2/'release.json').exists()


def test_preparation_pins_sources_and_never_starts_actor(tmp_path,monkeypatch):
    from linked_round_profile import prepare,verify
    socket_root=tmp_path/'sockets';evidence=tmp_path/'evidence'
    import subprocess
    def forbidden(*a,**kw):raise AssertionError('dispatched')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    prepared=prepare({'socket_root':str(socket_root)},evidence)
    verify(prepared)
    assert (evidence/'glm/task').is_dir() and (evidence/'qwen/task').is_dir()
    assert (evidence/'code/scripts/omp/paused_echo.mjs').exists()
    assert (evidence/'code/drift/serving/worker_owner_socket.py').exists()
    path=evidence/'code/scripts/omp/linked_round.py';path.write_text('tampered')
    with pytest.raises(ValueError):verify(prepared)
