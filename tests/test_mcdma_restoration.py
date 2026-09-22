"""Restore consecutive GLM requests through actual mailbox framing, without SSH payloads."""
import hashlib
import json
import time

import pytest

from drift.serving.mcdma_mailbox import HEADER_AT, Reader
from drift.serving.mcdma_reverse import ReversePublisher
from tests.test_mcdma_mailbox import FakeRegion, _bridge_module
from test_glm_restore import body
from test_glm_snapshot_bank import update
from test_glm_snapshot_binding import bound_session


class ImmediateRegion(FakeRegion):
    def put(self, data, offset=0):
        super().put(data, offset)
        if offset == HEADER_AT and hasattr(self, 'bridge'):
            payload = self.bridge.reader.peek()
            if payload is not None:
                self.bridge.handle(payload)
                self.bridge.reader.acknowledge()


def transport(tmp_path):
    regions, bridges = {}, {}
    for rank in range(2):
        region = ImmediateRegion()
        bridge = _bridge_module().Bridge(Reader(region), region, str(tmp_path / str(rank) / 'in'),
                                        str(tmp_path / str(rank) / 'out'), rank, rank == 0)
        region.bridge = bridge
        regions[rank], bridges[rank] = region, bridge
    return ReversePublisher(regions, head=0, timeout_s=.05), bridges


def test_staging_does_not_release_head_and_only_its_owner_can_release(tmp_path):
    publisher, bridges = transport(tmp_path)
    staged = publisher.stage('request-one', 0, b'fixture')
    folder = tmp_path / '0' / 'in' / 'request-one'
    assert (folder / '.000000.tmp.npz').read_bytes() == b'fixture'
    assert not (folder / '000000.npz').exists()
    delivered = publisher.release(staged)
    assert (folder / '000000.npz').read_bytes() == b'fixture'
    assert delivered['sha256'] == hashlib.sha256(b'fixture').hexdigest()
    with pytest.raises(Exception):
        publisher.release(staged)


def test_foreign_or_mutated_stage_never_releases_head(tmp_path):
    publisher, _ = transport(tmp_path)
    staged = publisher.stage('request-one', 0, b'fixture')
    with pytest.raises(Exception):
        publisher.release({**staged, 'session': 'request-two'})
    assert not (tmp_path / '0' / 'in' / 'request-one' / '000000.npz').exists()


def test_two_snapshot_versions_restore_via_mcdma_and_receipts_precede_output(tmp_path, monkeypatch):
    from drift.serving.mcdma_restoration import McdmaRestorationFactory
    session, bank, root, _ = bound_session(tmp_path, monkeypatch)
    publisher, bridges = transport(tmp_path)
    factory = McdmaRestorationFactory(publisher, layouts={'l3': (512,)}, max_rows=4)
    session.restoration.publication_factory = factory
    names = []

    def stream(request, timeout):
        name = request['kv_transfer_params']['drift_session']
        names.append(name)
        rows = request['kv_transfer_params']['drift_reserve']
        for rank, bridge in bridges.items():
            path = tmp_path / str(rank) / 'in' / name / '000000.npz'
            assert path.is_file()
            receipt = dict(rank=rank, world_size=2, sequence=0, rows=rows,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            out = tmp_path / str(rank) / 'out' / name
            out.mkdir(parents=True)
            (out / f'ack.000000.rank{rank}.json').write_text(json.dumps(receipt))
            bridge.relay_receipts()
        yield {'choices': [{'delta': {'content': 'own output'}}]}

    session.transport.inner.stream = stream
    for version in (1, 2):
        if version == 2:
            bank.publish(update(root, version, rows=3))
        request = {**body(), **session.prepare_turn(body())}
        session.transport.count_tokens(request, 1)
        assert list(session.transport.stream(request, 1))
        report = session.restoration.report()['turns'][-1]
        assert report['snapshot']['version'] == version
        assert len(report['receipts']) == 2
        assert {receipt['sha256'] for receipt in report['receipts']} == {bank.capture().sha256}
    assert len(set(names)) == 2


def test_bad_publication_never_touches_region(tmp_path):
    from drift.serving.mcdma_restoration import McdmaRestorationFactory

    publisher, _ = transport(tmp_path)
    factory = McdmaRestorationFactory(publisher, layouts={'l3': (512,)}, max_rows=4)
    path = tmp_path / 'invalid.npz'
    path.write_bytes(b'not an activation')
    before = [len(region.puts) for region in publisher.raw.values()]
    with pytest.raises(Exception):
        factory(path, hashlib.sha256(path.read_bytes()).hexdigest(), 'request-one', lambda: 1)
    assert [len(region.puts) for region in publisher.raw.values()] == before


def test_partial_rank_stage_never_releases_head_and_poison_prevents_retry(tmp_path):
    from drift.serving.mcdma_mailbox import MailboxError
    publisher, _ = transport(tmp_path)
    original = publisher._deliver
    def fail(rank, payload):
        if rank == 1:
            raise TimeoutError('synthetic loss')
        original(rank, payload)
    publisher._deliver = fail
    with pytest.raises(MailboxError, match='nothing was released'):
        publisher.stage('request-one', 0, b'fixture')
    assert not (tmp_path / '0' / 'in' / 'request-one' / '000000.npz').exists()
    publisher._deliver = original
    with pytest.raises(MailboxError):
        publisher.stage('request-two', 0, b'fixture')


def test_restore_requires_both_cache_receipts_after_mailbox_ack(tmp_path):
    from drift.serving.mcdma_restoration import McdmaRestorationFactory
    from drift.serving.mcdma_mailbox import MailboxError
    import numpy as np

    publisher, _ = transport(tmp_path)
    path = tmp_path / 'valid.npz'
    np.savez(path, l3=np.zeros((2, 512), np.float32))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    publication = McdmaRestorationFactory(publisher, layouts={'l3': (512,)}, max_rows=4)(path, digest, 'request-one', lambda: 1)
    publication.stage(timeout=1)
    publication.admit(0, digest)
    publication.publish(timeout=1)
    with pytest.raises(MailboxError):
        publication.wait_applied(0, 2, digest, deadline=time.monotonic() + 1)
    assert publication.failed


def test_nested_transport_deadline_keeps_parent_abort_check(tmp_path):
    from drift.exchange.lifetime import request_scope
    from drift.serving.mcdma_restoration import RestorationPublication

    publisher, _ = transport(tmp_path)
    def cancelled():
        raise RuntimeError('parent cancelled')
    publication = RestorationPublication(publisher, 'request-one', b'fixture', 'a' * 64, 2, lambda: 1)
    with request_scope(cancelled), pytest.raises(RuntimeError, match='parent cancelled'):
        publication.stage(timeout=1)
    assert publication.failed
    assert not (tmp_path / '0' / 'in').exists()


@pytest.mark.parametrize('timeout', [0, -1, True, float('nan'), float('inf')])
def test_expired_stage_is_rejected_before_wire_write(tmp_path, timeout):
    from drift.serving.mcdma_restoration import RestorationPublication

    publisher, _ = transport(tmp_path)
    publication = RestorationPublication(publisher, 'request-one', b'fixture', 'a' * 64, 2, lambda: 1)
    with pytest.raises(TimeoutError):
        publication.stage(timeout=timeout)
    assert publication.failed
    assert not (tmp_path / '0' / 'in').exists()


def test_empty_nested_scope_cannot_clear_parent_cancellation():
    from drift.exchange.lifetime import checkpoint, request_scope
    def cancelled():
        raise RuntimeError('cancelled')
    with request_scope(cancelled), request_scope(None), pytest.raises(RuntimeError, match='cancelled'):
        checkpoint()


@pytest.mark.parametrize('change', ['digest', 'rank_type', 'missing_rank', 'rows'])
def test_restore_receipt_preserves_the_strict_native_delivery_contract(tmp_path, change):
    from drift.serving.mcdma_restoration import RestorationPublication

    publisher, _ = transport(tmp_path)
    digest = hashlib.sha256(b'fixture').hexdigest()
    publication = RestorationPublication(publisher, 'request-one', b'fixture', digest, 2, lambda: 1)
    publication.stage(timeout=1)
    publication.admit(0, digest)
    publication.publish(timeout=1)
    receipts = {rank: dict(rank=rank, world_size=2, sequence=0, rows=2, sha256=digest) for rank in range(2)}
    if change == 'digest':
        receipts[0]['sha256'] = 'b' * 64
    elif change == 'rank_type':
        receipts[0]['rank'] = False
    elif change == 'missing_rank':
        receipts.pop(1)
    else:
        receipts[0]['rows'] = 1
    publisher.confirm_applied = lambda *args: {'receipts': receipts}
    with pytest.raises(ValueError, match='MCDMA_RESTORE_RECEIPT'):
        publication.wait_applied(0, 2, digest)
    assert publication.failed


def test_two_versions_collect_forward_rows_before_the_next_native_request(tmp_path, monkeypatch):
    import numpy as np
    from drift.serving.mcdma_restoration import McdmaRestorationFactory
    from drift.serving.mcdma_turn_outbox import McdmaTurnOutbox
    session, bank, root, _ = bound_session(tmp_path, monkeypatch)
    publisher, bridges = transport(tmp_path)
    forward_region = FakeRegion()
    reader = Reader(forward_region, session=91)
    bridges[0].forward_conn = forward_region
    exports = tmp_path / 'exports'
    exports.mkdir(mode=0o700)
    config = dict(memory_root=str(exports), source_worker='glm', target_worker='qwen', recipe_sha256='a'*64,
                  max_raw_rows=32, max_raw_bytes=65536, max_rows=8, max_bytes=32768, max_publications=4)
    session.restoration.publication_factory = McdmaRestorationFactory(publisher, layouts={'l3': (512,)}, max_rows=4)
    session.restoration.outbox = McdmaTurnOutbox(config, {'l3': (512,)}, reader, publisher,
                                               pause=lambda _: bridges[0].publish_taps())
    def stream(request, timeout):
        name = request['kv_transfer_params']['drift_session']
        assert request['kv_transfer_params']['drift_tap'] is True
        assert bridges[0].watching == name
        rows = request['kv_transfer_params']['drift_reserve']
        for rank, bridge in bridges.items():
            path = tmp_path / str(rank) / 'in' / name / '000000.npz'
            receipt = dict(rank=rank, world_size=2, sequence=0, rows=rows,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            out = tmp_path / str(rank) / 'out' / name
            out.mkdir(parents=True)
            (out / f'ack.000000.rank{rank}.json').write_text(json.dumps(receipt))
            if rank == 0:
                start, stop = 3 + rows, 12 + rows
                np.savez(out / '000000.npz', start=np.array(start), stop=np.array(stop),
                         l3=np.ones((stop-start, 512), dtype=np.float16))
                (out / 'finished').write_text(json.dumps(dict(failed='', writes_scheduled=1, tap_count=1,
                                                             source_start=start, source_stop=stop)))
            bridge.relay_receipts()
        yield {'choices': [{'delta': {'content': 'own output'}}]}
    session.transport.inner.stream = stream
    for version in (1, 2):
        if version == 2: bank.publish(update(root, version, rows=3))
        request = {**body(), **session.prepare_turn(body())}
        session.transport.count_tokens(request, 1)
        assert list(session.transport.stream(request, 1))
        turn = session.restoration.report()['turns'][-1]
        assert turn['outbox']['transport'] == 'mcdma' and turn['outbox']['selected_rows'] == 2
        assert len(turn['receipts']) == 2 and turn['snapshot']['version'] == version
        assert bridges[0].forward.acknowledged()
    assert reader.expected == 5
