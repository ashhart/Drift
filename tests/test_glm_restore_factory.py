"""Require pinned memory and route configuration before restored backend dispatch."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest


def setup(tmp_path, monkeypatch, mode='no_link_reserved'):
    import drift.serving.glm_restore_factory as module
    source = tmp_path / 'snapshot.npz'
    np.savez(source, l0=np.zeros((2, 512), dtype=np.float32))
    config = {'memory_mode': mode, 'pins': {'model_id': 'fixture'},
              'restoration': {'snapshot_path': str(source), 'snapshot_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                              'rows': 2, 'layers': [0], 'max_rows': 2, 'source_worker': 'qwen', 'target_worker': 'glm'}}
    session = SimpleNamespace(transport=SimpleNamespace(), started=10, limits={'deadline_ms': 5000, 'max_input_bytes': 4096})
    monkeypatch.setattr(module, 'unlinked_factory', lambda _: session)
    return module, config, session


def test_factory_constructs_fixed_deadline_reserved_control_without_route(tmp_path, monkeypatch):
    module, config, session = setup(tmp_path, monkeypatch)
    assert module.factory(config) is session
    assert session.restoration.deadline() == 15
    session.started = 12
    assert session.restoration.deadline() == 17
    assert session.restoration.linked is False


@pytest.mark.parametrize('mutation', ['hash', 'rows', 'layers', 'symlink'])
def test_factory_rejects_invalid_snapshot_before_dispatch(tmp_path, monkeypatch, mutation):
    module, config, _ = setup(tmp_path, monkeypatch)
    recipe = config['restoration']
    if mutation == 'hash': recipe['snapshot_sha256'] = '0' * 64
    if mutation == 'rows': recipe['rows'] = 3
    if mutation == 'layers': recipe['layers'] = [0, 1]
    if mutation == 'symlink':
        link = tmp_path / 'alias.npz'; link.symlink_to(recipe['snapshot_path']); recipe['snapshot_path'] = str(link)
    with pytest.raises(ValueError): module.factory(config)


def test_route_rejects_changed_pinned_host_file(tmp_path):
    from drift.serving.glm_restore_route import PinnedRoute
    identity, hosts = tmp_path / 'id', tmp_path / 'hosts'
    identity.write_text('fixture'); identity.chmod(0o600); hosts.write_text('fixture')
    profile = {'peer': '192.0.2.50', 'hostname': 'spark-b.invalid', 'identity_file': str(identity),
               'known_hosts_file': str(hosts), 'known_hosts_sha256': hashlib.sha256(hosts.read_bytes()).hexdigest()}
    path = tmp_path / 'route.json'; path.write_text(json.dumps(profile))
    route = PinnedRoute(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    hosts.write_text('changed')
    with pytest.raises(ValueError): route.validate()


def test_outbox_is_explicit_and_cannot_be_enabled_on_reserved_no_link(tmp_path,monkeypatch):
    module,config,_=setup(tmp_path,monkeypatch)
    config['outbox']={'source_worker':'glm','target_worker':'qwen'}
    with pytest.raises(ValueError):module.factory(config)
