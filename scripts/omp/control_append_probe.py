"""Measure additive controls through actual OMP and WorkerSession using public API fixtures."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch
import development_fixture


def qualify(omp, source, duo, python, port, snapshot_control=False):
    copy, run = shutil.copyfile, subprocess.run
    counters = {}

    def copied(src, dst, *args, **kwargs):
        result = copy(src, dst, *args, **kwargs)
        path = Path(dst)
        if Path(src).name == 'worker_context.ts' and snapshot_control:
            value = path.read_text()
            if 'await client.ownControlAppend(additions)' not in value:
                raise ValueError('CONTROL_FIXTURE_SOURCE')
            path.write_text(value.replace('await client.ownControlAppend(additions)', 'await client.ownControl(system)'))
        if Path(src).name == 'duo_backend_fixture.py':
            value = path.read_text().replace("        self.record('open')", "        self.initial_system = list(payload['system_prompt'])\n        self.facts['initial_control_characters'] = len('\\n\\n'.join(self.initial_system))\n        self.record('open')")
            path.write_text(value)
        if Path(src).name == 'worker_backend_fixture.py':
            value = path.read_text()
            for operation in ('own_control', 'own_control_append'):
                signature = f'    def {operation}(self, payload):\n'
                injection = signature + f"        self.facts['{operation}_characters'] = self.facts.get('{operation}_characters', 0) + len('\\n\\n'.join(payload['system_prompt']))\n"
                if operation == 'own_control_append':
                    injection += "        self.facts['replayed_initial_blocks'] = self.facts.get('replayed_initial_blocks', 0) + sum(value in getattr(self, 'initial_system', []) for value in payload['system_prompt'])\n"
                if signature not in value:
                    raise ValueError('CONTROL_FIXTURE_SOURCE')
                value = value.replace(signature, injection)
            path.write_text(value)
        return result

    def measured(*args, **kwargs):
        result = run(*args, **kwargs)
        config = json.loads(Path(kwargs['env']['DRIFT_WORKER_CONFIG']).read_text())
        for worker in config['workers']:
            backend = json.loads(Path(worker['command']['args'][4]).read_text())
            path = Path(backend['fixture_report'])
            facts = json.loads(path.read_text()) if path.exists() else {}
            counters[worker['identity']['worker']] = {key: facts.get(key, 0) for key in ('initial_control_characters', 'own_control', 'own_control_characters', 'own_control_append', 'own_control_append_characters', 'replayed_initial_blocks')}
        return result

    with patch.object(shutil, 'copyfile', copied), patch.object(subprocess, 'run', measured):
        report = development_fixture.run(omp, source, duo, python, port)
    report['control_counters'] = counters
    report['snapshot_control'] = snapshot_control
    report['executable_sha256'] = development_fixture.digest(omp.resolve())
    if not counters or not any(value['own_control_append'] for value in counters.values()) or any(value['own_control'] or value['replayed_initial_blocks'] for value in counters.values()):
        report['verdict'] = 'FAILED'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'))
    parser.add_argument('--port', type=int, default=49273)
    parser.add_argument('--snapshot-control', action='store_true')
    args = parser.parse_args()
    report = qualify(args.omp, args.source, args.duo, args.python, args.port, args.snapshot_control)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
