"""Qualify declared linked labels and fixed sessions against actual OMP without activations."""
import argparse
import json
from pathlib import Path
import shutil
from duo_setup import prepare_duo
from duo_probe import run as run_duo
from registration_setup import digest


def prepare_declared(root, source, duo):
    owner = prepare_duo(root, source, duo)
    helper = root/'scripts/omp/declared_worker_fixture.py'
    shutil.copyfile(Path(__file__).with_name(helper.name), helper)
    config = json.loads(owner.read_text())
    for entry in config['workers']:
        entry.update(memory_mode='linked', communication_mode='text_and_artifacts', session_binding='fixed')
        worker_path = Path(entry['command']['args'][4]); worker = json.loads(worker_path.read_text())
        worker['fixed_session'] = entry['identity']['session']; worker_path.write_text(json.dumps(worker))
        entry['command']['args'] = ['-S', '-m', 'scripts.omp.declared_worker_fixture', '--config', str(worker_path)]
        entry['artifacts'].append(dict(path=str(helper), sha256=digest(helper)))
        for artifact in entry['artifacts']: artifact['sha256'] = digest(Path(artifact['path']))
    owner.write_text(json.dumps(config))
    return owner


def run(omp, source, duo):
    report = run_duo(omp, source, duo, setup=prepare_declared)
    verified = all(report['facts'][name].get('fixed_session_verified') is True and report['facts'][name].get('activation_operations') == 0 for name in ('glm-fixture-worker', 'qwen-fixture-worker'))
    report.update(qualification='synthetic-declared-text-fixed-session', activation_exchange=False, fixed_sessions_verified=verified)
    if not verified: report['verdict'] = 'FAILED'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); report = run(args.omp, args.source, args.duo)
    print(json.dumps(report, sort_keys=True)); raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
