"""Run actual OMP and unchanged Duo with synthetic workers using the real scoped API tools."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from development_stage import prepare
from duo_setup import prepare_duo
from native_room_profile import digest


def run(omp, source, duo, python, port=49273):
    with tempfile.TemporaryDirectory(prefix='development-duo-fixture-') as directory:
        root = Path(directory).resolve(); fixture = root/'fixture'; fixture.mkdir()
        owner = prepare_duo(fixture, source, duo)
        for name in ('development_backend.py', 'api_fixture.py'):
            shutil.copyfile(Path(__file__).with_name(name), fixture/'scripts/omp'/name)
        config = json.loads(owner.read_text())
        for worker in config['workers']:
            worker['command']['args'][-1] = 'scripts.omp.development_backend:DevelopmentFixture'
            worker['artifacts'].extend(dict(path=str(fixture/'scripts/omp'/name), sha256=digest(fixture/'scripts/omp'/name)) for name in ('development_backend.py', 'api_fixture.py'))
        owner.write_text(json.dumps(config))
        profile = dict(owner=dict(path=str(owner), sha256=digest(owner)), duo=dict(path=str(duo)), omp=dict(path=str(omp)), parent='drift-experimental/glm-fixture', child='drift-experimental/qwen-fixture')
        prepared = prepare(profile, root/'evidence', python, port)
        result = subprocess.run(prepared['command'], cwd=prepared['cwd'], env=prepared['env'], capture_output=True, timeout=20)
        path = root/'evidence/facts.json'; facts = json.loads(path.read_text()) if path.exists() else {}
        passed = result.returncode == 0 and facts.get('public_verify_passes', 0) > 0 and facts.get('last_verified_api_sha256') == digest(Path(prepared['cwd'])/'api.py') and not facts.get('errors') and not facts.get('blocked') and facts.get('task_calls') == 1 and all(item.get('shutdown') for item in facts.get('workers', {}).values()) and len(facts.get('workers', {})) == 2
        stats_path = fixture/'qwen-fixture-worker.json'
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        counters = {key: stats.get(key) for key in ('api_step', 'api_result', 'skipped', 'errors', 'stream')}
        return dict(backend_counters=counters, verdict='PASSED' if passed else 'FAILED', qualification='synthetic-restricted-api-fixture', exit_code=result.returncode, facts=facts, stdout_bytes=len(result.stdout), stderr_bytes=len(result.stderr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'))
    parser.add_argument('--port', type=int, default=49273)
    args = parser.parse_args(); report = run(args.omp, args.source, args.duo, args.python, args.port)
    print(json.dumps(report, sort_keys=True)); raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
