"""Exercise the fixture controller against the shared supervisor using harmless local children."""
import os
from pathlib import Path
import sys
import time

import pytest

from drift.serving.fixture_host import claim_fixture
from drift.serving.fixture_supervision import run_fixture_child

ROOT=Path(__file__).resolve().parents[1]


def test_fixture_child_completion_is_reaped_and_output_discarded(monkeypatch):
    monkeypatch.setenv('PYTHONPATH',str(ROOT))
    receipt=run_fixture_child([sys.executable,'-c','print("PRIVATE_SYNTHETIC_OUTPUT")'],
                             deadline=time.monotonic()+12,work_seconds=7)
    assert receipt['reason']=='child_exit' and receipt['child_reaped'] and not receipt['process_group_alive']
    assert receipt['returncode']==receipt['supervisor_returncode']==0
    assert receipt['output_bytes']>0 and 'PRIVATE_SYNTHETIC_OUTPUT' not in str(receipt)
    with pytest.raises(ProcessLookupError): os.kill(receipt['child_pid'],0)


def test_fixture_deadline_stops_and_reaps_child_group(monkeypatch):
    monkeypatch.setenv('PYTHONPATH',str(ROOT))
    started=time.monotonic()
    receipt=run_fixture_child([sys.executable,'-c','import time; time.sleep(30)'],
                             deadline=started+12,work_seconds=7)
    assert receipt['reason']=='deadline' and receipt['child_reaped'] and not receipt['process_group_alive']
    assert receipt['returncode']!=0 and time.monotonic()-started<10


def test_private_claim_refuses_repository_and_open_permissions(tmp_path):
    repo=tmp_path/'repository'
    repo.mkdir(); (repo/'.git').write_text('synthetic worktree')
    with pytest.raises(ValueError,match='outside repositories'): claim_fixture(repo/'raw')
    public=tmp_path/'public'
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError,match='private'): claim_fixture(public)
    private=tmp_path/'private'
    first,second=claim_fixture(private),claim_fixture(private)
    assert first!=second and first.stat().st_mode & 0o077 == 0
    assert list((first/'raw').iterdir())==[]
