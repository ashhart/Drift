"""Run one explicitly bounded GLM cancellation probe on its serving host."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from scripts.qualify_live.cancel_driver import drive


def validate_profile(profile):
    fixed = {'version': 1, 'base_url': 'http://127.0.0.1:8888', 'max_requests': 1,
             'no_link': True, 'retries': 0, 'warmup_requests': 0,
             'baseline_samples': 3, 'settled_samples': 3, 'cleanup_reserve_seconds': 10}
    if any(type(profile.get(key)) is not type(value) or profile.get(key) != value for key, value in fixed.items()):
        raise ValueError('profile changes the cancellation-only scope')
    for key, maximum in [('max_new_tokens', 256), ('max_prompt_tokens', 256), ('reserve_rows', 8)]:
        value = profile.get(key)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('profile exceeds count limit')
    for key, minimum, maximum in [('max_seconds', 11, 180), ('poll_seconds', 0.05, 1)]:
        value = profile.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError('profile exceeds time limit')
    if profile.get('service_auth') not in {'none', 'bearer'}:
        raise ValueError('profile must declare the observed service authentication mode')
    if profile.get('model') != 'GLM-5.3-Flash-EXL3' or profile.get('engine') != '0':
        raise ValueError('profile target differs from the inspected engine')
    if not isinstance(profile.get('messages'), list) or len(json.dumps(profile['messages']).encode()) > 4096:
        raise ValueError('profile messages exceed byte limit')
    return profile


def validate_service_auth(profile, environment):
    mode = profile.get('service_auth')
    present = bool(environment.get('DRIFT_GLM_KEY'))
    if mode not in {'none', 'bearer'} or present != (mode == 'bearer'):
        raise ValueError('host credential does not match declared service authentication')


def main():
    from drift.serving.live_cancel_observation import CancellationObserver
    from drift.serving.live_metrics import parse_metrics
    from drift.serving.live_metric_fetch import fetch_metrics
    from drift.serving.live_qualification_client import Limits, OwnedSession, claim_session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    profile = validate_profile(json.loads(raw))
    validate_service_auth(profile, os.environ)
    with args.report.open('x') as report:
        claim = claim_session(args.ledger)
        messages = claim.path / 'messages.json'
        messages.write_text(json.dumps(profile['messages']))
        work_seconds = profile['max_seconds'] - profile['cleanup_reserve_seconds']
        limits = Limits(max_new=profile['max_new_tokens'], max_prompt_tokens=profile['max_prompt_tokens'],
                        max_seconds=work_seconds)
        command = [sys.executable, str(args.runner.resolve()), '--session', claim.name,
                   '--messages', str(messages.resolve()), '--max-new', str(limits.max_new),
                   '--reserve', str(profile['reserve_rows']), '--base', profile['base_url'],
                   '--model', profile['model'], '--no-link', '--control-stdin']
        client = OwnedSession(command, claim, limits)
        observer = CancellationObserver(timeout=work_seconds,
            baseline_samples=profile['baseline_samples'], settled_samples=profile['settled_samples'])
        def sample(*, timeout):
            text = fetch_metrics(profile['base_url'] + '/metrics', timeout=min(timeout, 5))
            return parse_metrics(text, model=profile['model'], engine=profile['engine'])
        result = drive(observer, client, sample, poll_seconds=profile['poll_seconds'],
                       max_seconds=profile['max_seconds'])
        result.update(profile_sha256=hashlib.sha256(raw).hexdigest(), no_link=True,
                      max_requests=1, actual_generated_tokens=None, warmup_requests=0, retries=0)
        report.write(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'PASSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
