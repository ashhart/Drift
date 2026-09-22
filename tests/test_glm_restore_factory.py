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


def test_mcdma_factory_never_constructs_ssh_route_or_payload_transport(tmp_path, monkeypatch):
    module, config, session = setup(tmp_path, monkeypatch, mode='linked_snapshot')
    config['restoration_transport'] = 'mcdma'
    def forbidden(*args, **kwargs):
        pytest.fail('SSH payload path must not be constructed')
    monkeypatch.setattr(module, 'PinnedRoute', forbidden)
    monkeypatch.setattr(module, 'PublicationTransport', forbidden)
    calls = []
    def publication(*args):
        calls.append(args)
        return 'runtime-owned transport'
    result = module.factory(config, publication_factory=publication, memory_root=tmp_path)
    assert result.restoration.root == tmp_path
    assert result.restoration.publication_factory is publication
    assert result.restoration.publication('fresh-request') == 'runtime-owned transport'
    assert calls[0][1:3] == (config['restoration']['snapshot_sha256'], 'fresh-request')


@pytest.mark.parametrize('declared, supplied', [('mcdma', False), ('ssh', True), ('unknown', False)])
def test_transport_declaration_cannot_select_a_fallback(tmp_path, monkeypatch, declared, supplied):
    module, config, _ = setup(tmp_path, monkeypatch, mode='linked_snapshot')
    config['restoration_transport'] = declared
    with pytest.raises(ValueError, match='transport must match'):
        module.factory(config, publication_factory=(lambda *args: None) if supplied else None)


def test_owner_bank_keeps_runtime_mcdma_factory_without_ssh_binding(tmp_path, monkeypatch):
    from test_glm_owner_worker import configuration
    from drift.serving.glm_owner_config import create_owner
    import drift.serving.glm_restore_factory as module

    root = tmp_path.resolve()
    root.chmod(0o700)
    config = configuration(root)
    config['restoration_transport'] = 'mcdma'
    session = SimpleNamespace(transport=SimpleNamespace(), started=None, poisoned=False,
                              limits=config['limits'])
    monkeypatch.setattr(module, 'unlinked_factory', lambda _: session)
    def forbidden(*args, **kwargs):
        pytest.fail('native owner must not construct an SSH route')
    monkeypatch.setattr(module, 'PinnedRoute', forbidden)
    publication = lambda *args: None
    backend, bank = create_owner(config, publication_factory=publication, memory_root=root, route_factory=forbidden)
    assert backend.snapshot_bank is bank
    assert backend.restoration.route is None
    assert backend.restoration.publication_factory is publication


def test_runtime_forward_factory_is_carried_into_owner_bank(tmp_path, monkeypatch):
    from test_glm_owner_worker import configuration
    from drift.serving.glm_owner_config import create_owner
    import drift.serving.glm_restore_factory as module
    root = tmp_path.resolve()
    root.chmod(0o700)
    config = configuration(root)
    config['restoration_transport'] = 'mcdma'
    config['outbox'] = {'source_worker': 'glm', 'target_worker': 'qwen'}
    session = SimpleNamespace(transport=SimpleNamespace(), started=None, poisoned=False, limits=config['limits'])
    monkeypatch.setattr(module, 'unlinked_factory', lambda _: session)
    calls, collector = [], SimpleNamespace(begin=lambda *args: None, finish=lambda *args: None, close=lambda: None)
    def forward(spec, layouts, *, max_turns):
        calls.append((spec, layouts, max_turns))
        return collector
    backend, bank = create_owner(config, publication_factory=lambda *args: None,
                                 outbox_factory=forward, memory_root=root)
    assert backend.restoration.outbox is collector and backend.snapshot_bank is bank
    assert calls == [(config['outbox'], {'l3': (512,)}, 2)]


@pytest.mark.parametrize('mode', ['ssh', 'missing_spec', 'missing_factory', 'wrong_owner'])
def test_forward_factory_cannot_bypass_transport_or_ownership(tmp_path, monkeypatch, mode):
    module, config, _ = setup(tmp_path, monkeypatch, mode='linked_snapshot')
    config['restoration_transport'] = 'mcdma'
    config['outbox'] = {'source_worker': 'glm', 'target_worker': 'qwen'}
    config['limits'] = {'max_turns': 2}
    kwargs = dict(publication_factory=lambda *args: None, outbox_factory=lambda *args, **kw: object())
    if mode == 'ssh':
        config['restoration_transport'] = 'ssh'
        kwargs['publication_factory'] = None
    elif mode == 'missing_spec': del config['outbox']
    elif mode == 'missing_factory': kwargs['outbox_factory'] = None
    else: config['outbox']['target_worker'] = 'other'
    with pytest.raises(ValueError): module.factory(config, **kwargs)
