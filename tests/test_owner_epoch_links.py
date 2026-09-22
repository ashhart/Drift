"""A staged native snapshot is not an applied receipt until the owner finishes its turn."""
import hashlib
import io
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from drift.exchange.coordinator import dispatch
from drift.exchange.owner_links import SnapshotOwnerLink, AppendOwnerLink
from drift.serving.glm_owner_socket import SnapshotSocket
from test_exchange_session import build
from test_glm_snapshot_bank import make_bank, update


def test_real_bank_stage_and_completed_receipt_advance_only_after_resume(tmp_path):
    bank, root = make_bank(tmp_path); bank.publish(update(root, 1))
    restorer = SimpleNamespace(boundary=threading.RLock(), failed=False, completed=[])
    endpoint = object.__new__(SnapshotSocket); endpoint.bank, endpoint.restorer = bank, restorer
    peer = SimpleNamespace(request=endpoint._dispatch)
    link = SnapshotOwnerLink(peer, root, session=bank.session, source_worker=bank.source,
                             target_worker=bank.target, recipe=bank.recipe, ranks=('rank0','rank1'))
    value = io.BytesIO(); np.savez(value, l3=np.ones((2,512), dtype=np.float32))
    body = value.getvalue(); digest = hashlib.sha256(body).hexdigest()
    exchange = build(); exchange.session = bank.session; exchange.ranks = ('rank0','rank1'); exchange.copies = 1
    exchange.source.snapshot = lambda _: dict(body=body, sha256=digest, rows=2)
    exchange.link = link
    reply = dispatch({'r':exchange}, {'op':'stage','route':'r'})
    assert reply['delivery']['state'] == 'STAGED_NOT_APPLIED' and exchange.sequence == 0
    assert endpoint._dispatch(dict(op='applied', version=2, sha256=digest))['state'] == 'PENDING'
    restorer.completed.append(dict(snapshot=dict(version=2, sha256=digest), snapshot_sha256=digest,
        receipts=[dict(rank=r, world_size=2, sequence=0, rows=2, sha256=digest) for r in (0,1)]))
    reply = dispatch({'r':exchange}, {'op':'confirm','route':'r','sequence':0})
    assert reply['published']['receipts'] == [digest,digest] and exchange.sequence == 1


def test_staging_cannot_overwrite_pending_or_accept_partial_rank_application(tmp_path):
    bank, root = make_bank(tmp_path); initial = update(root, 1); bank.publish(initial)
    restorer = SimpleNamespace(boundary=threading.RLock(), failed=False, completed=[dict(
        snapshot=dict(version=1,sha256=initial['sha256']),snapshot_sha256=initial['sha256'],receipts=[])])
    endpoint = object.__new__(SnapshotSocket); endpoint.bank, endpoint.restorer = bank, restorer
    with pytest.raises(ValueError): endpoint._dispatch(dict(op='applied',version=1,sha256=initial['sha256']))
    session = build(); session.link.stage = session.link.deliver
    dispatch({'r':session},dict(op='stage',route='r'))
    with pytest.raises(RuntimeError): dispatch({'r':session},dict(op='stage',route='r'))
    assert session.poisoned and len(session.link.delivered) == 1


def test_append_refuses_wrong_owner_identity_even_with_matching_digest(tmp_path):
    peer = SimpleNamespace(requests=1)
    def request(frame):
        return dict(v=1,session='wrong',seq=1,source_worker='glm',target_worker='qwen',
                    op='appended',sha256=frame['sha256'],rows=frame['rows'],foreign_total=frame['rows'])
    peer.request = request
    link = AppendOwnerLink(peer,tmp_path,rank='qwen',session='s',source_worker='glm',target_worker='qwen')
    value=io.BytesIO(); np.savez(value,k3=np.ones((2,1,512),dtype=np.float32))
    with pytest.raises(ValueError): link.stage('s',0,value.getvalue(),1)
    assert link.sequence == 0


@pytest.mark.parametrize('bad', ['session', 'boolean_sequence', 'boolean_rows'])
def test_append_rejects_misbound_or_untyped_evidence(tmp_path, bad):
    calls = []
    def request(frame):
        calls.append(frame)
        reply = dict(v=1,session='s',seq=1,source_worker='glm',target_worker='qwen',
                     op='appended',sha256=frame['sha256'],rows=1,foreign_total=1)
        if bad == 'boolean_sequence': reply['seq'] = True
        if bad == 'boolean_rows': reply['rows'] = True
        return reply
    peer = SimpleNamespace(requests=1, request=request)
    link = AppendOwnerLink(peer,tmp_path,rank='qwen',session='s',source_worker='glm',target_worker='qwen')
    value=io.BytesIO(); np.savez(value,k3=np.ones((1,1,512),dtype=np.float32))
    with pytest.raises(ValueError): link.stage('wrong' if bad == 'session' else 's',0,value.getvalue(),1)
    assert link.sequence == 0
    if bad == 'session': assert calls == []
