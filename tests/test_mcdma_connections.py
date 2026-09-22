"""Admit every dedicated target before writing the forward mailbox."""
import time
from types import SimpleNamespace

import pytest

from drift.serving.mcdma_mailbox import Reader, Window
from tests.test_mcdma_mailbox import FakeRegion


def fixture():
    regions, targets, opened = [], [], []
    for rank in range(2):
        region = FakeRegion()
        Reader(Window(region, 0, region.region_len // 2), session=100 + rank)
        region.puts.clear()
        regions.append(region)
        targets.append(dict(rank=rank, peer=f'192.0.2.{rank+1}', source=f'192.0.2.{rank+10}',
                            region_len=region.region_len, region_vaddr=4096, rkey=rank+1, session=100+rank))
    def opener(peer, src):
        rank = next(index for index, item in enumerate(targets) if item['peer'] == peer)
        region = regions[rank]
        conn = SimpleNamespace(region_len=region.region_len, region_vaddr=4096, rkey=rank+1,
                               get=region.get, put=region.put, closed=False)
        def close():
            conn.closed = True
        conn.close = close
        opened.append(conn)
        return conn
    return dict(v=1, targets=targets), regions, opened, opener


def test_admission_opens_separate_forward_connection_and_closes_all():
    from drift.serving.mcdma_connections import McdmaConnections
    config, regions, opened, opener = fixture()
    with McdmaConnections(config, opener, deadline=time.monotonic()+2) as runtime:
        assert tuple(runtime.publisher.conns) == (0, 1)
        assert runtime.reader.expected == 1
        assert len(opened) == 3
        assert regions[0].puts and not regions[1].puts
        assert not any(conn.closed for conn in opened)
    assert all(conn.closed for conn in opened)


@pytest.mark.parametrize('field', ['region_len', 'region_vaddr', 'rkey', 'session'])
def test_any_wrong_target_refuses_before_first_write_and_closes_opened(field):
    from drift.serving.mcdma_connections import McdmaConnections
    config, regions, opened, opener = fixture()
    config['targets'][1][field] += 1
    with pytest.raises(ValueError):
        McdmaConnections(config, opener, deadline=time.monotonic()+2)
    assert all(conn.closed for conn in opened)
    assert not any(region.puts for region in regions)


def test_wrong_head_on_second_open_refuses_before_first_write():
    from drift.serving.mcdma_connections import McdmaConnections
    config, regions, opened, opener = fixture()
    def bad(peer, src):
        conn = opener(peer, src)
        if len(opened) == 3:
            conn.rkey += 1
        return conn
    with pytest.raises(ValueError):
        McdmaConnections(config, bad, deadline=time.monotonic()+2)
    assert all(conn.closed for conn in opened)
    assert not any(region.puts for region in regions)


def test_open_failure_closes_previously_opened_connections():
    from drift.serving.mcdma_connections import McdmaConnections
    config, regions, opened, opener = fixture()
    def fail(peer, src):
        if opened:
            raise TimeoutError('synthetic open failure')
        return opener(peer, src)
    with pytest.raises(TimeoutError):
        McdmaConnections(config, fail, deadline=time.monotonic()+2)
    assert len(opened) == 1 and opened[0].closed
    assert not any(region.puts for region in regions)


@pytest.mark.parametrize('deadline', [0, True, float('nan'), float('inf')])
def test_bad_deadline_never_opens_a_connection(deadline):
    from drift.serving.mcdma_connections import McdmaConnections
    config, _, opened, opener = fixture()
    with pytest.raises((ValueError, TimeoutError)):
        McdmaConnections(config, opener, deadline=deadline)
    assert not opened


def test_closed_owner_cannot_write_again():
    from drift.serving.mcdma_connections import McdmaConnections
    config, regions, _, opener = fixture()
    runtime = McdmaConnections(config, opener, deadline=time.monotonic()+2)
    runtime.close()
    counts = [len(region.puts) for region in regions]
    with pytest.raises(Exception):
        runtime.publisher.watch('new-request')
    assert [len(region.puts) for region in regions] == counts
