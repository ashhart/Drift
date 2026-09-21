"""Rank admission and receipts use synthetic files and CPU cache fixtures only."""
import hashlib
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from drift.serving.live_rank_delivery import PROBE, RankDelivery, RankDeliveryError
from test_vllm_glm53_inject import connector, LAYERS

DIGEST = 'a' * 64


@pytest.mark.parametrize('fault', ['error', 'missing', 'digest', 'no_ack', 'wrong_rank', 'wrong_world', 'wrong_rows'])
def test_every_rank_must_admit_and_acknowledge(fault):
    checked = []

    def ssh(host, command, *, timeout=None):
        request = json.loads(shlex.split(command)[-1])
        rank = request['rank']
        checked.append((host, rank))
        ack = {'rank': rank, 'world_size': 2, 'sequence': 0, 'rows': 3, 'sha256': DIGEST}
        status = {'error': False, 'present': True, 'sha256': DIGEST, 'ack': ack}
        if rank == 1:
            if fault == 'error': status['error'] = True
            if fault == 'missing': status['present'] = False
            if fault == 'digest': status['sha256'] = 'b' * 64
            if fault == 'no_ack': status['ack'] = None
            if fault == 'wrong_rank': ack['rank'] = 0
            if fault == 'wrong_world': ack['world_size'] = 1
            if fault == 'wrong_rows': ack['rows'] = 2
        return json.dumps(status)

    monitor = RankDelivery(('head', 'peer'), 'synthetic', ssh, 0.001, lambda: None)
    with pytest.raises(RankDeliveryError):
        monitor.admit(0, DIGEST)
        monitor.wait_applied(0, 3, DIGEST)
    assert ('head', 0) in checked and ('peer', 1) in checked


@pytest.mark.parametrize('malformed_ack', [False, True])
def test_actual_probe_reports_only_fixed_status_fields(tmp_path, malformed_ack):
    incoming, outgoing = tmp_path / 'tp-live-in' / 'synthetic', tmp_path / 'tp-live-out' / 'synthetic'
    incoming.mkdir(parents=True)
    outgoing.mkdir(parents=True)
    payload = incoming / '000000.npz'
    payload.write_bytes(b'synthetic activation bytes')
    (outgoing / 'error.rank1').write_text('private synthetic exception that must not be returned')
    ack = {'rank': 1, 'world_size': 2, 'sequence': 0, 'rows': 3, 'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}
    (outgoing / 'ack.000000.rank1.json').write_text(json.dumps({**ack, **({'private': 'private synthetic payload'} if malformed_ack else {})}))
    request = {'run': 'synthetic', 'rank': 1, 'sequence': 0, 'temporary': False}
    script = PROBE.replace('/dev/shm/glm53-handoff', str(tmp_path))
    result = subprocess.run([sys.executable, '-S', '-c', script, json.dumps(request)], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {'error': True, 'present': True, 'sha256': ack['sha256'], 'ack': {'invalid': True} if malformed_ack else ack}
    assert 'private synthetic' not in result.stdout


def live_step():
    return NS(name='rank-test', request_id='r', reserve=4, reserve_start=0, blocks=((0,), (1,)),
              before=4, after=5, apply=(0,), tap=False, failed='')


@pytest.mark.parametrize('rank', [0, 1])
def test_receipt_matches_actual_rank_bytes_and_completed_rows(connector, rank):
    c, root = connector
    c._tp = lambda: (rank, 2)
    incoming = root / 'tp-live-in' / 'rank-test'
    incoming.mkdir(parents=True)
    path = incoming / '000000.npz'
    np.savez(path, l3=np.ones((2, 512), np.float32), l7=np.ones((2, 512), np.float32))
    c.meta = NS(inject=[], live=[live_step()], requests=[])
    c.wait_for_save()
    receipt = root / 'tp-live-out' / 'rank-test' / f'ack.000000.rank{rank}.json'
    assert json.loads(receipt.read_text()) == {'rank': rank, 'world_size': 2, 'sequence': 0, 'rows': 2,
                                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert all(c._kv_caches[layer].any() for layer in LAYERS)
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    assert c._live_worker['rank-test']['poisoned']
    assert c._live_worker['rank-test']['filled'] == 2


def test_peer_missing_publication_writes_error_without_ack(connector):
    c, root = connector
    c._tp = lambda: (1, 2)
    c.meta = NS(inject=[], live=[live_step()], requests=[])
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    outgoing = root / 'tp-live-out' / 'rank-test'
    assert (outgoing / 'error.rank1').exists()
    assert not list(outgoing.glob('ack.*'))
    assert not any(c._kv_caches[layer].any() for layer in LAYERS)


def test_partial_local_write_cannot_produce_rank_ack(connector, monkeypatch):
    import torch
    c, root = connector
    c._tp = lambda: (1, 2)
    incoming = root / 'tp-live-in' / 'rank-test'
    incoming.mkdir(parents=True)
    np.savez(incoming / '000000.npz', l3=np.ones((2, 512), np.float32), l7=np.ones((2, 512), np.float32))
    original = torch.Tensor.__setitem__
    def write(tensor, index, value):
        if tensor is c._kv_caches[LAYERS[1]]:
            raise RuntimeError('synthetic device failure')
        return original(tensor, index, value)
    monkeypatch.setattr(torch.Tensor, '__setitem__', write)
    c.meta = NS(inject=[], live=[live_step()], requests=[])
    with pytest.raises(RuntimeError, match='poisoned'):
        c.wait_for_save()
    outgoing = root / 'tp-live-out' / 'rank-test'
    assert (outgoing / 'error.rank1').exists()
    assert not list(outgoing.glob('ack.*'))
    assert c._live_worker['rank-test']['poisoned']
    assert c._kv_caches[LAYERS[0]].any() and not c._kv_caches[LAYERS[1]].any()


@pytest.mark.parametrize('delay', [1.0, 5.0])
def test_valid_ack_at_or_after_deadline_is_rejected(monkeypatch, delay):
    now = [0.0]
    def ssh(host, command, **kwargs):
        now[0] = delay
        return json.dumps({'error': False, 'present': True, 'sha256': DIGEST,
            'ack': {'rank': 0, 'world_size': 1, 'sequence': 0, 'rows': 3, 'sha256': DIGEST}})
    monkeypatch.setattr('drift.serving.live_rank_delivery.time.monotonic', lambda: now[0])
    monitor = RankDelivery(('head',), 'synthetic', ssh, 1.0, lambda: None)
    with pytest.raises(RankDeliveryError, match='deadline'):
        monitor.wait_applied(0, 3, DIGEST)


def test_all_rank_probes_share_one_remaining_timeout(monkeypatch):
    now, timeouts = [0.0], []
    def ssh(host, command, **kwargs):
        request = json.loads(shlex.split(command)[-1])
        timeouts.append(kwargs.get('timeout'))
        now[0] += 0.2
        return json.dumps({'error': False, 'present': True, 'sha256': DIGEST,
            'ack': {'rank': request['rank'], 'world_size': 3, 'sequence': 0, 'rows': 3, 'sha256': DIGEST}})
    monkeypatch.setattr('drift.serving.live_rank_delivery.time.monotonic', lambda: now[0])
    monitor = RankDelivery(('head', 'peer1', 'peer2'), 'synthetic', ssh, 1.0, lambda: None)
    monitor.wait_applied(0, 3, DIGEST)
    assert timeouts == pytest.approx([1.0, 0.8, 0.6])


def test_expired_failure_check_prevents_starting_another_probe(monkeypatch):
    now = [0.0]
    def check_failure():
        now[0] = 2.0
    def ssh(*args, **kwargs):
        pytest.fail('expired acknowledgement budget still launched SSH')
    monkeypatch.setattr('drift.serving.live_rank_delivery.time.monotonic', lambda: now[0])
    monitor = RankDelivery(('head',), 'synthetic', ssh, 1.0, check_failure)
    with pytest.raises(RankDeliveryError, match='deadline'):
        monitor.wait_applied(0, 3, DIGEST)
