"""Exercise bounded multi-turn own control through installed OMP and real worker core."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from probe import probe_environment, sandbox_policy
from registration_setup import digest, prepare


def run(omp, source):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='drift-lifetime-') as directory:
        root = Path(directory).resolve()
        owner = prepare(root, source)
        config = json.loads(owner.read_text())
        entry = config['workers'][0]
        entry['experimental_multi_turn'] = True
        worker_path = Path(entry['command']['args'][4])
        worker = json.loads(worker_path.read_text())
        worker['experimental_multi_turn'] = True
        worker_path.write_text(json.dumps(worker))
        entry['command']['args'][-1] = 'scripts.omp.worker_backend_fixture:LifetimeFixture'
        for artifact in entry['artifacts']:
            artifact['sha256'] = digest(Path(artifact['path']))
        config['workers'] = [entry]
        owner.write_text(json.dumps(config))
        fixture = root / 'scripts/omp/lifetime_fixture.mjs'
        shutil.copyfile(Path(__file__).with_name(fixture.name), fixture)
        policy = root / 'policy.sb'
        policy.write_text(sandbox_policy(root, Path.home()))
        (root / 'agent').mkdir()
        env = probe_environment(root) | {'DRIFT_WORKER_CONFIG': str(owner), 'DRIFT_WORKER_CONFIG_SHA256': digest(owner)}
        command = ['/usr/bin/sandbox-exec', '-f', str(policy), str(omp.resolve()), '--cwd', str(root), '--no-session', '--no-tools', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(root / 'scripts/omp/experimental_extension.mjs'), '--extension', str(fixture), '--model', 'drift-experimental/glm-fixture', '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '20', 'First own prompt.', 'Second own prompt.']
        result = subprocess.run(command, env=env, cwd=root, capture_output=True, timeout=30)
        path = root / 'glm-fixture-worker.json'
        facts = json.loads(path.read_text()) if path.exists() else {}
        passed = result.returncode == 0 and facts.get('open') == 1 and facts.get('own_prompt') == 2 and facts.get('stream') == 2 and facts.get('own_control') == 1
        return dict(verdict='PASSED' if passed else 'BLOCKED', exit_code=result.returncode, facts=facts, seconds=round(time.monotonic() - started, 3), executable_sha256=digest(omp.resolve()), worker_session_sha256=digest(root / 'drift/serving/worker_session.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=Path, required=True)
    parser.add_argument('--worker-source', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.omp, args.worker_source)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
