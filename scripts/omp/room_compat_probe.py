"""Reproduce installed OMP multicall and Hub compatibility using public deterministic fixtures."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
from duo_probe import run


def fixture_source(source, multicall):
    original = "elif turn == 3 or getattr(self, 'skipped', False):\n                call = ('hub', {'op': 'wait', 'ids': ['DuoPeer'], 'timeoutMs': 2000})"
    replacement = "elif turn == 3:\n                call = ('hub', {'op': 'list'})\n            elif turn == 4:\n                call = ('hub', {'op': 'wait'})\n            elif turn == 5:\n                call = ('hub', {'op': 'wait', 'from': 'DuoPeer'})"
    if original not in source:
        raise ValueError('fixture source changed')
    source = source.replace(original, replacement)
    if multicall:
        original = "call = ('todo', {'op': 'view'})"
        replacement = "yield {'op': 'tool_call', 'payload': {'call_id': 'todo-first', 'name': 'todo', 'arguments': {'op': 'view', 'i': 'fixture intent'}}}\n                call = ('task', {'agent': 'duo-peer', 'name': 'DuoPeer', 'context': 'Synthetic fixture', 'tasks': [{'agent': 'duo-peer', 'name': 'DuoPeer', 'task': 'Send PROPOSE fixture to Main through Hub.'}]})"
        if original not in source:
            raise ValueError('fixture source changed')
        source = source.replace(original, replacement).replace('elif turn == 2:', 'elif turn == 99:')
        source = source.replace('elif turn == 3:', 'elif turn == 2:').replace('elif turn == 4:', 'elif turn == 3:').replace('elif turn == 5:', 'elif turn == 98:')
    return source


def qualify(omp, source, duo, multicall=False):
    copy = shutil.copyfile

    def copied(src, dst, *args, **kwargs):
        result = copy(src, dst, *args, **kwargs)
        path = Path(dst)
        if Path(src).name == 'duo_backend_fixture.py':
            path.write_text(fixture_source(path.read_text(), multicall))
        if Path(src).name == 'duo_fixture.mjs':
            source = path.read_text()
            observer = '  api.on("tool_result", event => { if (event.toolName === "hub" && event.details?.op === "list") { const peers = event.details.peers; facts.hub_list = { count: peers?.length ?? -1, scoped: Array.isArray(peers) && peers.every(peer => ["Main", "DuoPeer"].includes(peer.id)) }; save(); } });\n'
            path.write_text(source[:source.rfind('}')] + observer + source[source.rfind('}'):])
        return result

    with patch.object(shutil, 'copyfile', copied):
        report = run(omp, source, duo)
    roster = report['facts']['facts'].get('hub_list', {})
    if roster.get('scoped') is not True or not 0 <= roster.get('count', -1) <= 1:
        report['verdict'] = 'BLOCKED'
    report['fixture'] = 'multicall' if multicall else 'hub-forms'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=Path, required=True)
    parser.add_argument('--worker-source', type=Path, required=True)
    parser.add_argument('--duo', type=Path, required=True)
    parser.add_argument('--multicall', action='store_true')
    args = parser.parse_args()
    report = qualify(args.omp, args.worker_source, args.duo, args.multicall)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
