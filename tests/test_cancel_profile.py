"""Reject an expanded qualification workload before any process or request starts."""
import json
from pathlib import Path

import pytest

from scripts.qualify_live.cancel import validate_profile, validate_service_auth


@pytest.mark.parametrize('field,value', [('max_new_tokens', 257), ('max_prompt_tokens', 257),
    ('max_seconds', 181), ('max_requests', 2), ('retries', 1), ('warmup_requests', 1),
    ('no_link', False), ('reserve_rows', 9), ('poll_seconds', 0), ('max_new_tokens', True),
    ('base_url', 'http://example.com'), ('baseline_samples', 1), ('cleanup_reserve_seconds', 0),
    ('max_seconds', 10)])
def test_profile_cannot_expand_frozen_cancellation_scope(field, value):
    profile = json.loads(Path('configs/qualification.cancel-glm-v1.json').read_text())
    profile[field] = value
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_reviewed_profile_is_accepted():
    profile = json.loads(Path('configs/qualification.cancel-glm-v1.json').read_text())
    assert validate_profile(profile) is profile


@pytest.mark.parametrize('mode,key', [('none', ''), ('bearer', 'synthetic-key')])
def test_explicit_service_auth_accepts_matching_host_configuration(mode, key):
    validate_service_auth({'service_auth': mode}, {'DRIFT_GLM_KEY': key})


@pytest.mark.parametrize('mode,key', [('none', 'synthetic-key'), ('bearer', ''), ('missing', '')])
def test_service_auth_rejects_unexpected_configuration_without_exposing_key(mode, key):
    with pytest.raises(ValueError) as caught:
        validate_service_auth({'service_auth': mode}, {'DRIFT_GLM_KEY': key})
    assert 'synthetic-key' not in str(caught.value)
