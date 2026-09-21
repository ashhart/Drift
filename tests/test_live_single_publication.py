"""Single-publication mechanics do not establish model use or TP atomicity."""
import json
import shlex

import pytest

from drift.serving.live_rank_delivery import RankDelivery, RankDeliveryError

DIGEST = 'a' * 64


def test_admission_rejects_valid_results_after_shared_deadline(monkeypatch):
    now = [0.0]
    def ssh(host, command, **kwargs):
        now[0] = 2.0
        return json.dumps({'error': False, 'present': True, 'sha256': DIGEST, 'ack': None})
    monkeypatch.setattr('drift.serving.live_rank_delivery.time.monotonic', lambda: now[0])
    delivery = RankDelivery(('head',), 'one', ssh, 10, lambda: None)
    with pytest.raises(RankDeliveryError, match='deadline'):
        delivery.admit(0, DIGEST, deadline=1.0)


def test_verified_receipts_are_returned_for_private_aggregate_evidence():
    def ssh(host, command, **kwargs):
        rank = json.loads(shlex.split(command)[-1])['rank']
        return json.dumps({'error': False, 'present': True, 'sha256': DIGEST,
            'ack': {'rank': rank, 'world_size': 2, 'sequence': 0, 'rows': 12, 'sha256': DIGEST}})
    delivery = RankDelivery(('head', 'peer'), 'one', ssh, 1, lambda: None)
    receipts = delivery.wait_applied(0, 12, DIGEST)
    assert receipts == [{'rank': rank, 'world_size': 2, 'sequence': 0, 'rows': 12, 'sha256': DIGEST} for rank in (0, 1)]


def test_one_publication_orders_all_rank_admission_before_release_and_retains_receipts():
    from drift.serving.live_single_publication import SinglePublication
    events = []
    def ssh(host, command, **kwargs):
        request = json.loads(shlex.split(command)[-1])
        events.append((host, request['sequence'], request['temporary']))
        rank = request['rank']
        return json.dumps({'error': False, 'present': True, 'sha256': DIGEST,
            'ack': {'rank': rank, 'world_size': 2, 'sequence': 0, 'rows': 12, 'sha256': DIGEST}})
    delivery = RankDelivery(('head', 'peer'), 'one', ssh, 1, lambda: None)
    gate = SinglePublication(delivery, rows=12, digest=DIGEST, timeout=1)
    def stage(*, timeout): events.append('stage')
    def release(*, timeout):
        assert events[-2:] == [('head', 0, True), ('peer', 0, False)]
        events.append('release')
    report = gate.run(stage, release)
    assert report['status'] == 'PASSED' and len(report['receipts']) == 2
    assert report['staged'] == report['released'] == 1
    assert report['worker_shutdown'] == 'BLOCKED'
    with pytest.raises(RuntimeError, match='reused'): gate.run(stage, release)


def test_no_link_performs_no_transport_or_release():
    from drift.serving.live_single_publication import SinglePublication
    def forbidden(**kwargs): pytest.fail('no-link invoked transport')
    gate = SinglePublication(None, rows=12, digest=DIGEST, timeout=1, no_link=True)
    report = gate.run(forbidden, forbidden)
    assert report['status'] == 'PASSED' and report['rows'] == report['released'] == 0
    assert report['scope'] == 'no_link_transport' and report['receipts'] == []


def test_failed_peer_admission_never_releases_and_hides_exception_payload():
    from drift.serving.live_single_publication import SinglePublication
    def ssh(host, command, **kwargs):
        request = json.loads(shlex.split(command)[-1])
        return json.dumps({'error': False, 'present': host == 'head', 'sha256': DIGEST, 'ack': None})
    delivery = RankDelivery(('head', 'peer'), 'one', ssh, 1, lambda: None)
    gate = SinglePublication(delivery, rows=12, digest=DIGEST, timeout=1)
    report = gate.run(lambda **kwargs: None, lambda **kwargs: pytest.fail('released absent peer data'))
    assert report['status'] == 'INVALID' and report['phase'] == 'ADMISSION'
    assert report['released'] == 0 and report['error_type'] == 'RankDeliveryError'


def test_late_stage_cannot_admit_or_release(monkeypatch):
    from drift.serving.live_single_publication import SinglePublication
    now = [0.0]
    def ssh(host, command, **kwargs):
        return json.dumps({'error': False, 'present': True, 'sha256': DIGEST, 'ack': None})
    monkeypatch.setattr('drift.serving.live_rank_delivery.time.monotonic', lambda: now[0])
    delivery = RankDelivery(('head',), 'one', ssh, 1, lambda: None)
    gate = SinglePublication(delivery, rows=12, digest=DIGEST, timeout=1, clock=lambda: now[0])
    def stage(**kwargs): now[0] = 2
    report = gate.run(stage, lambda **kwargs: pytest.fail('late release'))
    assert report['status'] == 'INVALID' and report['phase'] == 'STAGING' and report['released'] == 0
