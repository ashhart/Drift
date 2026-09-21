"""Prepare pinned synthetic manifests against copied actual worker protocol modules."""
import hashlib
import json
from pathlib import Path
import shutil
from provider_boundary_setup import copy_boundary_modules


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, source):
    repo = Path(__file__).resolve().parents[2]
    for relative in ('scripts/omp', 'plugin/omp-drift/src', 'drift/serving'):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for path in (repo / 'plugin/omp-drift/src').glob('worker_*.ts'):
        shutil.copyfile(path, root / 'plugin/omp-drift/src' / path.name)
    for name in ('experimental_extension.mjs', 'registration_fixture.mjs', 'worker_backend_fixture.py'):
        shutil.copyfile(Path(__file__).with_name(name), root / 'scripts/omp' / name)
    copy_boundary_modules(root / 'scripts/omp')
    for path in (source / 'drift/serving').glob('worker_*.py'):
        shutil.copyfile(path, root / 'drift/serving' / path.name)
    for path in ('drift/__init__.py', 'drift/serving/__init__.py'):
        (root / path).write_text('')
    artifact = root / 'artifact.txt'
    artifact.write_text('fixture only; no model weights')
    manifests = {}
    for kind in ('model', 'translator'):
        path = root / (kind + '.json')
        path.write_text(json.dumps({'artifacts': {'fixture': str(artifact)}, 'sha256': {'fixture': digest(artifact)}, 'weights_verification': 'fixture: no weights'}))
        manifests[kind] = path
    limits = dict(max_input_bytes=65536, max_output_tokens=128, max_session_tokens=1024, max_turns=2, deadline_ms=5000)
    workers = []
    for name in ('glm-fixture', 'qwen-fixture'):
        pins = dict(model_id=name, model_sha256=digest(manifests['model']), translator_sha256=digest(manifests['translator']))
        config = root / (name + '.json')
        config.write_text(json.dumps(dict(worker=name, pins=pins, limits=limits, model_manifest_path=str(manifests['model']), translator_manifest_path=str(manifests['translator']), fixture_report=str(root / (name + '-worker.json')))))
        workers.append(dict(identity=dict(session=name, worker=name, **pins), limits=limits, expected=dict(backend='fixture', nativeStates=['fixture']), context_window=8192, memory_mode='no-link',
            command=dict(executable='/usr/bin/python3', sha256=digest('/usr/bin/python3'), args=['-S', '-m', 'drift.serving.worker_stdio', '--config', str(config), '--backend', 'scripts.omp.worker_backend_fixture:FixtureBackend'], cwd=str(root), env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'}),
            artifacts=[dict(path=str(config), sha256=digest(config)), dict(path=str(root / 'scripts/omp/worker_backend_fixture.py'), sha256=digest(root / 'scripts/omp/worker_backend_fixture.py'))] + [dict(path=str(path), sha256=digest(path)) for path in sorted((root / 'drift/serving').glob('worker_*.py'))]))
    owner = root / 'owner.json'
    owner.write_text(json.dumps(dict(version=1, experimental=True, workers=workers)))
    return owner
