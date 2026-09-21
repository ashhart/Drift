"""Prepare or explicitly run the restricted no-link two-agent public API fixture."""
import argparse
import json
from pathlib import Path
import time
from native_room_profile import load, digest
from native_room_stage import verify_stage
from native_room_evidence import read_cleanup, room_ok
from native_room_process import execute
from development_stage import prepare


def run(args):
    if args.arm != 'text_only': raise ValueError('DEVELOPMENT_LINKED_UNQUALIFIED')
    profile, owner = load(args.profile, args.profile_sha256, development=True)
    extended = profile['purpose'] == 'restricted-api-development'
    evidence = args.evidence.resolve(); ready = evidence/'development-prepared.json'
    if not args.run:
        prepared = prepare(profile, evidence, args.python, args.port); prepared['profile_sha256'] = args.profile_sha256
        ready.write_text(json.dumps(prepared, sort_keys=True, indent=2))
        return dict(verdict='PREPARED', qualification='restricted-api-development', prepared_sha256=digest(ready), profile_sha256=args.profile_sha256)
    if not args.prepared_sha256 or digest(ready) != args.prepared_sha256: raise ValueError('DEVELOPMENT_PREPARED_PIN')
    prepared = json.loads(ready.read_text())
    if prepared['profile_sha256'] != args.profile_sha256: raise ValueError('DEVELOPMENT_PROFILE_PIN')
    verify_stage(prepared)
    if any(digest(path) != expected for path, expected in prepared['source_pins'].items()): raise ValueError('DEVELOPMENT_SOURCE_PIN')
    for item in profile['cleanup']: read_cleanup(item, fresh=True)
    (evidence/'development.started').open('x').close()
    started = time.monotonic(); result = {}; cleanup = []; errors = []
    try: result = execute(prepared['command'], cwd=prepared['cwd'], env=prepared['env'], **({'timeout': 174} if extended else {}))
    except Exception: errors.append('CONTROLLER_FAILURE')
    finally:
        for item in profile['cleanup']:
            try: cleanup.append(read_cleanup(item, fresh=False, **({'max_seconds': 180} if extended else {})))
            except Exception: errors.append('CLEANUP_UNVERIFIED')
    facts_path = evidence/'facts.json'
    facts = json.loads(facts_path.read_text()) if facts_path.exists() and facts_path.stat().st_size <= 65536 else {}
    token_limits = {'drift-experimental/' + entry['identity']['model_id']: entry['limits']['max_session_tokens'] for entry in owner['workers']}
    orchestration = result.get('exit_code') == 0 and result.get('output_joined') and not any(result.get(key) for key in ('timed_out', 'output_limit', 'descendants_killed')) and not errors and len(cleanup) == 2 and all(item['passed'] for item in cleanup) and room_ok(facts, [profile['parent'], profile['child']], token_limits, max_turns=10 if extended else 6)
    api_sha256 = digest(Path(prepared['cwd'])/'api.py')
    report = dict(verdict='PASSED' if orchestration and facts.get('public_verify_passes', 0) > 0 and facts.get('last_verified_api_sha256') == api_sha256 else 'FAILED', qualification='restricted-api-development', channels=prepared['channels'], controller=result, facts=facts, cleanup=cleanup, errors=errors, profile_sha256=args.profile_sha256, prepared_sha256=digest(ready), api_sha256=api_sha256, seconds=time.monotonic()-started)
    (evidence/'development-report.json').write_text(json.dumps(report, sort_keys=True, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('profile', 'evidence'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profile-sha256', required=True); parser.add_argument('--prepared-sha256')
    parser.add_argument('--python', type=Path, default=Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'))
    parser.add_argument('--port', type=int, required=True); parser.add_argument('--arm', choices=['text_only', 'activation_supplement'], default='text_only')
    parser.add_argument('--run', action='store_true'); report = run(parser.parse_args())
    print(json.dumps(report, sort_keys=True)); raise SystemExit(0 if report['verdict'] in ('PREPARED', 'PASSED') else 2)
