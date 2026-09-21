"""Require the explicit single-publication profile before any live client starts."""
import json
from pathlib import Path

import pytest

from scripts.qualify_live.linked import validate_profile


@pytest.mark.parametrize('field,value', [('no_link', True), ('linked', False), ('max_publications', True),
    ('max_publications', 2), ('publication_rows', 13), ('reserve_rows', 13), ('export_taps', True),
    ('max_new_tokens', 257), ('max_prompt_tokens', 257), ('peer', 'host; false'), ('max_seconds', 181)])
def test_linked_profile_cannot_expand_scope(field, value):
    profile = json.loads(Path('configs/qualification.linked-glm-v1.json').read_text())
    profile[field] = value
    with pytest.raises(ValueError): validate_profile(profile)


def test_prepared_linked_profile_is_valid():
    profile = json.loads(Path('configs/qualification.linked-glm-v1.json').read_text())
    assert validate_profile(profile) is profile
