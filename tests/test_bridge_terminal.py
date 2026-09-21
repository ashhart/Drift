"""Bind accepted-cache completion labels to the pinned source receipt chain."""
import json

import pytest

from test_translation_bridge import folders, forward, forward_reader, prefix_input


def complete(report):
    report['final_tail'] = 'VERIFIED_ACCEPTED_CACHE'
    report['native_terminal'] = dict(tap_count=2, source_start=5, source_stop=22)
    report['raw_publications'] = [dict(seq=0, sha256='a' * 64, bytes=40000, start=5, stop=20),
                                  dict(seq=1, sha256='b' * 64, bytes=40000, start=20, stop=22)]


def test_verified_native_frontier_survives_translation_receipt(tmp_path, monkeypatch):
    source, output = folders(tmp_path)
    manifest, report = prefix_input(source)
    complete(report)
    manifest.write_text(json.dumps(report))
    result = forward(source, output, manifest, forward_reader(monkeypatch))
    assert result['final_tail'] == 'VERIFIED_ACCEPTED_CACHE'
    assert result['native_terminal'] == report['native_terminal']
    assert result['full_completion'] is False


@pytest.mark.parametrize('fault', ['label_only', 'count', 'stop', 'start', 'gap',
    'source_seq', 'source_bounds', 'raw_bytes', 'raw_hash', 'legacy_evidence'])
def test_native_terminal_cannot_be_promoted_or_detached_from_rows(tmp_path, monkeypatch, fault):
    source, output = folders(tmp_path)
    manifest, report = prefix_input(source)
    complete(report)
    if fault == 'label_only':
        del report['native_terminal']
    elif fault == 'count':
        report['native_terminal']['tap_count'] = 1
    elif fault == 'stop':
        report['native_terminal']['source_stop'] = 23
    elif fault == 'start':
        report['native_terminal']['source_start'] = 4
    elif fault == 'gap':
        report['raw_publications'][1]['start'] = 19
    elif fault == 'source_seq':
        report['publications'][0]['source_seq'] = 0
    elif fault == 'source_bounds':
        report['raw_publications'][0]['stop'] = 21
        report['raw_publications'][1]['start'] = 21
    elif fault == 'raw_bytes':
        report['raw_publications'][0]['bytes'] = 1
    elif fault == 'raw_hash':
        report['raw_publications'][0]['sha256'] = 'invalid'
    else:
        report['final_tail'] = 'UNKNOWN'
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        forward(source, output, manifest, forward_reader(monkeypatch))
    assert not (output / 'forward.npz').exists()
