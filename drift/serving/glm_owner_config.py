"""Bind a separately pinned recipe and immutable initial snapshot to one GLM owner bank."""
from pathlib import Path
import re
from drift.serving.glm_restore_factory import ROOT, factory
from drift.serving.glm_restore_route import PinnedRoute, canonical_file
from drift.serving.glm_snapshot_bank import SnapshotBank, require
from drift.serving.glm_snapshot_binding import bind_snapshot_bank
from drift.serving.worker_activation_files import private_root
from drift.serving.worker_artifacts import digest


def control_paths(configuration):
    control = configuration['owner_control']
    require(type(control) is dict and set(control) == {'root', 'socket_name', 'evidence_name'})
    root = private_root(control['root'])
    require(root.stat().st_mode & 0o777 == 0o700)
    require(not any((path/'.git').exists() for path in (root, *root.parents)))
    for name in ('socket_name', 'evidence_name'):
        require(type(control[name]) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', control[name]))
    require(control['socket_name'] != control['evidence_name'])
    return root/control['socket_name'], root/control['evidence_name']


def create_owner(configuration, *, backend_factory=factory, route_factory=PinnedRoute,
                 publication_factory=None, outbox_factory=None, memory_root=ROOT):
    require(configuration['memory_mode'] == 'linked_snapshot')
    require(outbox_factory is None or publication_factory is not None)
    spec, restoration = configuration['owner_bank'], configuration['restoration']
    keys = {'root', 'session', 'recipe_path', 'recipe_sha256', 'max_rows', 'max_bytes', 'max_versions', 'max_total_bytes', 'initial'}
    require(type(spec) is dict and set(spec) == keys)
    root = private_root(spec['root'])
    require(not any((path/'.git').exists() for path in (root, *root.parents)))
    recipe = canonical_file(spec['recipe_path'], 1048576)
    require(digest(recipe, 1048576) == spec['recipe_sha256'])
    initial = spec['initial']
    require(type(initial) is dict and initial['version'] == 1 and initial['session'] == spec['session'])
    require(initial['sha256'] == restoration['snapshot_sha256'] and initial['rows'] == restoration['rows'])
    require(root/initial['path'] == Path(restoration['snapshot_path']))
    require(restoration['target_worker'] == configuration['worker'])
    require(spec['max_rows'] <= restoration['max_rows'])
    bank = SnapshotBank(root, session=spec['session'], source_worker=restoration['source_worker'],
                        target_worker=restoration['target_worker'], recipe_sha256=spec['recipe_sha256'],
                        layouts={f'l{layer}': (512,) for layer in restoration['layers']},
                        **{key: spec[key] for key in ('max_rows', 'max_bytes', 'max_versions', 'max_total_bytes')})
    backend = None
    try:
        bank.publish(initial)
        if publication_factory is None:
            backend = backend_factory(configuration)
            route = route_factory(restoration['route_path'], restoration['route_sha256'])
        else:
            options = {'outbox_factory': outbox_factory} if outbox_factory is not None else {}
            backend = backend_factory(configuration, publication_factory=publication_factory, memory_root=memory_root, **options)
            route = None
        bind_snapshot_bank(backend, bank, route, translator_sha256=spec['recipe_sha256'])
        return backend, bank
    except Exception:
        bank.close()
        if backend is not None: backend.close()
        raise
