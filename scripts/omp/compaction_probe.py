"""Exercise actual OMP compaction with deterministic public API workers and realistic usage."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import development_fixture


def run(omp, source, duo, python, port, enabled):
    copy = shutil.copyfile
    prepare = development_fixture.prepare_duo
    stage = development_fixture.prepare

    def prepared(root, source, duo):
        owner = prepare(root, source, duo); config = json.loads(owner.read_text())
        for worker in config['workers']:
            parent = worker['identity']['model_id'] == 'glm-fixture'
            worker['limits']['max_session_tokens'] = 65536 if parent else 20000
            worker['context_window'] = 16384 if parent else 8192
            path = Path(worker['command']['args'][4]); value = json.loads(path.read_text())
            value['limits'] = worker['limits']; path.write_text(json.dumps(value))
            for artifact in worker['artifacts']: artifact['sha256'] = development_fixture.digest(artifact['path'])
        owner.write_text(json.dumps(config)); return owner

    def staged(*args):
        value = stage(*args)
        if enabled:
            Path(value['env']['PI_CODING_AGENT_DIR'], 'config.yml').write_text('compaction:\n  enabled: true\n')
        return value

    def copied(src, dst, *args, **kwargs):
        value = copy(src, dst, *args, **kwargs); path = Path(dst)
        name = Path(src).name
        if name not in ('duo_backend_fixture.py', 'development_backend.py', 'development_extension.mjs'): return value
        text = path.read_text()
        if name == 'duo_backend_fixture.py':
            text = text.replace("'input_tokens': 3", "'input_tokens': (9214 if turn == 1 else 17303 if turn == 2 else 1000) if self.worker == 'glm-fixture' else (7001 if turn == 1 else 1000)")
        if name == 'development_backend.py':
            text = text.replace("'input_tokens': 3", "'input_tokens': 7001 if self.facts['stream'] == 1 else 1000")
        if name == 'development_extension.mjs':
            text = text.replace('  let command;', '  let command;\n  api.on("auto_compaction_start", () => { facts.compactions = (facts.compactions ?? 0) + 1; save(); });')
        if text == path.read_text(): raise ValueError('COMPACTION_FIXTURE_SOURCE')
        path.write_text(text); return value

    with patch.object(shutil, 'copyfile', copied), patch.object(development_fixture, 'prepare_duo', prepared), patch.object(development_fixture, 'prepare', staged):
        report = development_fixture.run(omp, source, duo, python, port)
    report.update(qualification='synthetic-retained-history-compaction', compaction_enabled=enabled)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo', 'python'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=49273); parser.add_argument('--enable-compaction', action='store_true')
    args = parser.parse_args(); report = run(args.omp, args.source, args.duo, args.python, args.port, args.enable_compaction)
    print(json.dumps(report, sort_keys=True)); raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
