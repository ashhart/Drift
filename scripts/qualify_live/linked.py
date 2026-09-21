"""Qualify one precomputed private publication on the head host without retaining model output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from scripts.qualify_live.cancel import validate_profile as validate_cancel_profile, validate_service_auth
from scripts.qualify_live.linked_driver import drive_linked


def validate_profile(profile):
    if profile.get('linked') is not True or profile.get('no_link') is not False or profile.get('reserve_rows') != 12 or profile.get('publication_rows') != 12:
        raise ValueError('linked qualification requires exactly twelve reserved publication rows')
    if type(profile.get('max_publications')) is not int or profile['max_publications'] != 1 or profile.get('export_taps') is not False:
        raise ValueError('linked qualification requires one publication without outbound taps')
    if not isinstance(profile.get('peer'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', profile['peer']):
        raise ValueError('linked qualification peer is invalid')
    validate_cancel_profile(dict(profile, no_link=True, reserve_rows=8))
    return profile


def main():
    from drift.serving.live_cancel_observation import CancellationObserver
    from drift.serving.live_linked_client import LinkedSession
    from drift.serving.live_metrics import parse_metrics
    from drift.serving.live_metric_fetch import fetch_metrics
    from drift.serving.live_publication import load_publication
    from drift.serving.live_publication_transport import PublicationTransport
    from drift.serving.live_qualification_client import Limits, claim_session
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'runner', 'ledger', 'report', 'publication'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--publication-sha256', required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    profile = validate_profile(json.loads(raw))
    validate_service_auth(profile, os.environ)
    if not re.fullmatch('[0-9a-f]{64}', args.publication_sha256): raise ValueError('invalid publication hash')
    if args.publication.stat().st_size > 1048576: raise ValueError('publication exceeds byte cap')
    layers = {f'l{3 + 4 * i}': (512,) for i in range(11)}
    arrays = load_publication(args.publication, layers, max_rows=12)
    if any(len(value) != 12 for value in arrays.values()): raise ValueError('publication row count differs')
    del arrays
    with args.report.open('x') as report:
        claim = claim_session(args.ledger)
        messages = claim.path / 'messages.json'
        messages.write_text(json.dumps(profile['messages']))
        work_seconds = profile['max_seconds'] - profile['cleanup_reserve_seconds']
        limits = Limits(max_new=profile['max_new_tokens'], max_prompt_tokens=profile['max_prompt_tokens'], max_seconds=work_seconds)
        command = [sys.executable, str(args.runner.resolve()), '--session', claim.name,
            '--messages', str(messages.resolve()), '--max-new', str(limits.max_new), '--reserve', '12',
            '--max-prompt-tokens', str(limits.max_prompt_tokens), '--base', profile['base_url'], '--model', profile['model'],
            '--control-stdin', '--wait-publication', '--no-tap', '--ready-timeout', str(work_seconds)]
        client = LinkedSession(command, claim, limits)
        def check_failure():
            client.poll(timeout=0)
            state = client.request_state()
            if state.completed or state.cancelled: raise RuntimeError('owned request ended before delivery')
        transport = PublicationTransport(args.publication, args.publication_sha256, claim.name, profile['peer'], check_failure)
        observer = CancellationObserver(timeout=work_seconds, baseline_samples=3, settled_samples=3)
        def sample(*, timeout):
            text = fetch_metrics(profile['base_url'] + '/metrics', timeout=min(timeout, 5))
            return parse_metrics(text, model=profile['model'], engine=profile['engine'])
        result = drive_linked(observer, client, sample, transport, rows=12, digest=args.publication_sha256,
                              max_seconds=profile['max_seconds'], poll_seconds=profile['poll_seconds'])
        result.update(profile_sha256=hashlib.sha256(raw).hexdigest(), publication_sha256=args.publication_sha256,
                      max_requests=1, warmup_requests=0, retries=0, actual_generated_tokens=None)
        report.write(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'PASSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
