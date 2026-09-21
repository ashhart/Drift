import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import pytest
from drift.serving.worker_activation_client import ActivationClient
from drift.serving.worker_activation_server import ActivationServer
from drift.serving.worker_owner_socket import OwnerSocket
from test_worker_own_snapshot import fixture


@pytest.mark.parametrize('change',[{'rows':2},{'scope':'foreign'},{'seq':True},{'session':'old'},
    {'source_worker':'glm'},{'target_worker':'qwen'},{'after':1},{'source_start':6999},
    {'source_stop':7000},{'available_own_rows':1000001},{'sha256':'bad'},
    {'selection':'all'},{'extra':'forbidden'}])
def test_client_rejects_invalid_selected_receipt_and_closes(change):
    own,peer=socket.socketpair();config=dict(session='s',source_worker='glm',target_worker='qwen',max_rows=4,max_total_rows=4)
    client=ActivationClient(own,config,time.monotonic()+1)
    def respond():
        peer.recv(4096)
        value=dict(v=1,session='s',seq=1,source_worker='qwen',target_worker='glm',op='own_snapshot',selection='last_own_row',scope='own',rows=1,after=0,source_start=7000,source_stop=7001,available_own_rows=7001,sha256='a'*64)
        value.update(change);peer.sendall(json.dumps(value).encode()+b'\n')
    thread=threading.Thread(target=respond);thread.start()
    try:
        with pytest.raises(RuntimeError,match='ACTIVATION_CHANNEL_FAILED'):client.snapshot_last_own('one.npz',0)
        assert client.socket.fileno()==-1 and client.selected_stop==0
    finally:thread.join(1);client.close();peer.close()


def test_owner_socket_routes_selected_snapshot_over_separate_activation_channel():
    with tempfile.TemporaryDirectory(prefix='snap-',dir='/tmp') as temp:
        root=Path(temp).resolve();control,config,state,calls=fixture(root)
        parent,child=socket.socketpair();failed=threading.Event()
        server=ActivationServer({**config,'fd':os.dup(child.fileno())},control.gate,control.layouts,failed.set)
        server.start();child.close();client=ActivationClient(parent,config,time.monotonic()+2)
        owner=OwnerSocket(root/'owner.sock',time.monotonic()+2);owner.start(client)
        peer=socket.socket(socket.AF_UNIX);peer.settimeout(1);peer.connect(str(root/'owner.sock'))
        try:
            peer.sendall(b'{"op":"snapshot_last_own","path":"one.npz","after":0}\n')
            receipt=json.loads(peer.recv(4096))
            assert receipt['op']=='own_snapshot' and receipt['source_start']==7000 and receipt['rows']==1
            assert owner.requests==1 and not failed.is_set() and calls[-1]=='settled'
            assert client.next_first==0 and client.selected_stop==7001
            owner.close()
        finally:
            peer.close();server.close();client.close();owner.close()
