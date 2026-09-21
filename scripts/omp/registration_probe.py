"""Verify both configured workers through actual OMP and actual Python session core."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
from probe import probe_environment, sandbox_policy
from registration_setup import digest, prepare


def run(omp, source):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='drift-registration-') as directory:
        root = Path(directory).resolve()
        owner = prepare(root, source)
        policy = root / 'policy.sb'
        policy.write_text(sandbox_policy(root, Path.home()))
        results = []
        for model in ('glm-fixture', 'qwen-fixture'):
            (root / 'agent').mkdir(exist_ok=True)
            env = probe_environment(root) | {'DRIFT_WORKER_CONFIG': str(owner), 'DRIFT_WORKER_CONFIG_SHA256': digest(owner)}
            command = ['/usr/bin/sandbox-exec', '-f', str(policy), str(omp.resolve()), '--cwd', str(root), '--no-session', '--no-tools', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(root / 'scripts/omp/experimental_extension.mjs'), '--extension', str(root / 'scripts/omp/registration_fixture.mjs'), '--model', 'drift-experimental/' + model, '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '20', 'Run the deterministic fixture.']
            result = subprocess.run(command, env=env, cwd=root, capture_output=True, timeout=40)
            worker = root / (model + '-worker.json')
            facts = json.loads(worker.read_text()) if worker.exists() else {}
            tools = json.loads((root / 'facts.json').read_text()) if (root / 'facts.json').exists() else {}
            passed = result.returncode == 0 and facts == dict(open=1, own_prompt=1, tool_result=1, stream=2, close=1) and tools == dict(tool_calls=1, tool_events=1, other_tools=0)
            results.append(dict(model=model, verdict='PASSED' if passed else 'BLOCKED', exit_code=result.returncode, worker=facts, tools=tools))
        return dict(verdict='PASSED' if all(row['verdict'] == 'PASSED' for row in results) else 'BLOCKED', results=results, seconds=round(time.monotonic() - started, 3), executable_sha256=digest(omp.resolve()), worker_session_sha256=digest(root / 'drift/serving/worker_session.py'), worker_stdio_sha256=digest(root / 'drift/serving/worker_stdio.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=Path, required=True)
    parser.add_argument('--worker-source', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = run(args.omp, args.worker_source)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
