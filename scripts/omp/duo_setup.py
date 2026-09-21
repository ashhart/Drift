"""Stage unmodified stock Duo and deterministic backend into an isolated probe directory."""
import json
from pathlib import Path
import shutil
from registration_setup import digest, prepare


def prepare_duo(root, source, duo):
    owner = prepare(root, source)
    package = root / 'scripts/omp/stock_duo'
    shutil.copytree(duo / 'src', package / 'src')
    for name in ('duo_fixture.mjs', 'duo_backend_fixture.py', 'room_guard.mjs', 'room_policy.mjs', 'room_diagnostics.mjs', 'room_hub.mjs'):
        shutil.copyfile(Path(__file__).with_name(name), root / 'scripts/omp' / name)
    for folder in (root / 'agents', root / '.omp/agents'):
        folder.mkdir(parents=True)
        shutil.copyfile(duo / 'agents/duo-peer.md', folder / 'duo-peer.md')
    config = json.loads(owner.read_text())
    for entry in config['workers']:
        entry['experimental_multi_turn'] = True
        entry['limits'].update(max_turns=6, max_input_bytes=262144, deadline_ms=15000)
        worker_path = Path(entry['command']['args'][4])
        worker = json.loads(worker_path.read_text())
        worker.update(experimental_multi_turn=True, limits=entry['limits'])
        worker_path.write_text(json.dumps(worker))
        entry['command']['args'][-1] = 'scripts.omp.duo_backend_fixture:DuoFixture'
        entry['artifacts'].append({'path': str(root / 'scripts/omp/duo_backend_fixture.py'), 'sha256': ''})
        for artifact in entry['artifacts']:
            artifact['sha256'] = digest(Path(artifact['path']))
    owner.write_text(json.dumps(config))
    return owner
