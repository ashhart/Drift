import hashlib
import json
import socket
import threading
import time
import numpy as np
import pytest
from drift.serving.worker_activation import ActivationControl
from drift.serving.worker_activation_client import ActivationClient
from drift.serving.worker_native_gate import NativeGate


def fixture(tmp_path, count=7001):
    root=tmp_path/'memory';root.mkdir(mode=0o700)
    state=dict(own_slots=list(range(count)),slot_pos=list(range(128,128+count))+[0,1],own_pos=128+count,reserve=128,cache=object())
    calls=[]
    def handle(command):
        calls.append(command)
        slots=state['own_slots'][command['first']:]
        np.savez(command['out'],k0=np.full((len(slots),1,2),slots[-1],np.float16),v0=np.full((len(slots),1,2),slots[-1],np.float16))
        return {'tapped':len(slots),'next_first':len(state['own_slots'])}
    gate=NativeGate(handle,state,lambda:calls.append('settled'))
    config=dict(session='s',source_worker='glm',target_worker='qwen',memory_root=str(root),max_bytes=65536,max_rows=4096,max_total_rows=4096)
    return ActivationControl(config,gate,{'k0':(1,2),'v0':(1,2)}),config,state,calls


def frame(seq=1,after=0,path='selected.npz'):
    return dict(v=1,session='s',seq=seq,op='snapshot_last_own',path=path,after=after)


def test_7001_own_rows_export_one_and_preserve_old_tap_cursor(tmp_path):
    control,config,state,calls=fixture(tmp_path)
    reply=control.apply(frame())
    assert reply==dict(v=1,session='s',seq=1,op='own_snapshot',source_worker='qwen',target_worker='glm',selection='last_own_row',scope='own',rows=1,after=0,source_start=7000,source_stop=7001,available_own_rows=7001,sha256=hashlib.sha256((control.root/'selected.npz').read_bytes()).hexdigest())
    assert calls[0]['first']==7000 and calls[-1]=='settled' and control.next_first==0
    with np.load(control.root/'selected.npz') as arrays:
        assert arrays['k0'].shape==(1,1,2) and arrays['k0'][0,0,0]==7000
    assert (control.root/'selected.npz').stat().st_mode&0o777==0o400


def test_legacy_tap_still_rejects_7001_rows(tmp_path):
    control,config,state,calls=fixture(tmp_path)
    with pytest.raises(RuntimeError):control.apply(dict(v=1,session='s',seq=1,op='tap',path='old.npz',first=0,max_rows=4096))
    assert calls==[] and state['poisoned']


@pytest.mark.parametrize('fault',['empty','foreign','stale','replay','budget','extra','session'])
def test_selected_snapshot_fails_closed_before_native_dispatch(tmp_path,fault):
    control,config,state,calls=fixture(tmp_path);request=frame()
    if fault=='empty':state['own_slots']=[]
    if fault=='foreign':state['own_slots'][-1]=len(state['slot_pos'])-1
    if fault=='stale':request['after']=7001
    if fault=='replay':request['seq']=0
    if fault=='budget':control.max_bytes=5
    if fault=='extra':request['text']='forbidden'
    if fault=='session':request['session']='other'
    with pytest.raises(RuntimeError):control.apply(request)
    assert not calls and state['poisoned'] and not (control.root/'selected.npz').exists()


def test_selected_snapshot_requires_new_own_rows_and_has_independent_cursor(tmp_path):
    control,config,state,calls=fixture(tmp_path)
    control.apply(frame())
    with pytest.raises(RuntimeError):control.apply(frame(2,7001,'again.npz'))
    assert len(calls)==2 and control.next_first==0


def test_selected_snapshot_uses_actual_socket_client_and_preserves_tap_cursor(tmp_path):
    control,config,state,calls=fixture(tmp_path);owned,peer=socket.socketpair()
    client=ActivationClient(owned,config,time.monotonic()+1)
    def respond():
        request=json.loads(peer.recv(4096));peer.sendall(json.dumps(control.apply(request)).encode()+b'\n')
    thread=threading.Thread(target=respond);thread.start()
    try:
        result=client.snapshot_last_own('selected.npz',0)
        assert result['source_stop']==7001 and client.selected_stop==7001 and client.next_first==0
    finally:thread.join(1);client.close();peer.close()


def test_second_snapshot_advances_without_delivering_skipped_rows(tmp_path):
    control,config,state,calls=fixture(tmp_path)
    control.apply(frame())
    state['own_slots'].extend(range(len(state['slot_pos']),len(state['slot_pos'])+3))
    state['slot_pos'].extend(range(state['own_pos'],state['own_pos']+3));state['own_pos']+=3
    reply=control.apply(frame(2,7001,'new.npz'))
    assert (reply['after'],reply['source_start'],reply['source_stop'],reply['rows'])==(7001,7003,7004,1)
    assert control.next_first==0 and control.appended==0


def test_selection_waits_for_native_chunk_and_settlement(tmp_path):
    control,config,state,calls=fixture(tmp_path)
    entered=threading.Event();release=threading.Event();output=[]
    def mutate():
        with control.gate.transaction():
            entered.set();release.wait(1)
            state['own_slots'].append(len(state['slot_pos']));state['slot_pos'].append(state['own_pos']);state['own_pos']+=1
    worker=threading.Thread(target=mutate);worker.start();assert entered.wait(1)
    reader=threading.Thread(target=lambda:output.append(control.apply(frame())));reader.start()
    time.sleep(.01);assert calls==[] and output==[]
    release.set();worker.join(1);reader.join(1)
    assert output[0]['source_start']==7001 and calls[-1]=='settled'
