"""Prepare by default, then explicitly run one frozen native no-link room qualification."""
import argparse
import json
from pathlib import Path
import time
from native_room_profile import load, digest
from native_room_stage import prepare, verify_stage
from native_room_evidence import read_cleanup, room_ok
from native_room_process import execute


def run(profile_path, profile_sha256, evidence, *, dispatch=False, prepared_sha256=None):
    profile, owner = load(profile_path, profile_sha256)
    evidence = evidence.resolve(); ready = evidence/'prepared.json'
    if not dispatch:
        prepared = prepare(profile, evidence); prepared['profile_sha256'] = profile_sha256
        ready.write_text(json.dumps(prepared, sort_keys=True, indent=2))
        return dict(verdict='PREPARED', prepared_sha256=digest(ready), profile_sha256=profile_sha256, evidence=str(evidence))
    if prepared_sha256 is None or digest(ready) != prepared_sha256: raise ValueError('ROOM_PREPARED_PIN')
    prepared = json.loads(ready.read_text())
    if prepared['profile_sha256'] != profile_sha256: raise ValueError('ROOM_PREPARED_PIN')
    verify_stage(prepared)
    for item in profile['cleanup']: read_cleanup(item, fresh=True)
    (evidence/'run.started').open('x').close()
    started = time.monotonic(); result = {}; cleanup = []; errors = []
    try: result = execute(prepared['command'], cwd=prepared['cwd'], env=prepared['env'])
    except Exception: errors.append('CONTROLLER_FAILURE')
    finally:
        for item in profile['cleanup']:
            try: cleanup.append(read_cleanup(item, fresh=False))
            except Exception: errors.append('CLEANUP_UNVERIFIED')
    facts_path = evidence/'facts.json'
    facts = json.loads(facts_path.read_text()) if facts_path.exists() and facts_path.stat().st_size <= 65536 else {}
    token_limits = {'drift-experimental/' + entry['identity']['model_id']: entry['limits']['max_session_tokens'] for entry in owner['workers']}
    passed = result.get('exit_code') == 0 and result.get('output_joined') and not result.get('timed_out') and not result.get('output_limit') and not result.get('descendants_killed') and not errors and len(cleanup) == 2 and all(item['passed'] for item in cleanup) and room_ok(facts, [profile['parent'], profile['child']], token_limits)
    report = dict(verdict='PASSED' if passed else 'FAILED', qualification='restricted-no-link-room', profile_sha256=profile_sha256, prepared_sha256=digest(ready), controller=result, facts=facts, cleanup=cleanup, errors=errors, seconds=time.monotonic()-started)
    (evidence/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True); parser.add_argument('--profile-sha256', required=True)
    parser.add_argument('--evidence', type=Path, required=True); parser.add_argument('--run', action='store_true'); parser.add_argument('--prepared-sha256')
    args = parser.parse_args()
    report = run(args.profile, args.profile_sha256, args.evidence, dispatch=args.run, prepared_sha256=args.prepared_sha256)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report['verdict'] in ('PREPARED', 'PASSED') else 2)
