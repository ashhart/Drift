import hashlib
import io
import json
import os
from types import SimpleNamespace
from threading import Event

import pytest

from drift.exchange.bootstrap import load_profile, serve


def profile(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / 'profile.json'
    path.write_text('{"v":1}')
    path.chmod(0o600)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('control', [b'{"op":"shutdown"}\n', b''])
def test_bootstrap_joins_cleanup_before_closed(tmp_path, control):
    path, pin = profile(tmp_path)
    events = []
    class Controller:
        server = SimpleNamespace(failed=Event())
        def __init__(self, value, deadline):
            assert value == {'v': 1}
        def start(self):
            events.append('started')
        def close(self):
            events.append('closed')
    read, write = os.pipe()
    os.write(write, control); os.close(write)
    output = io.StringIO()
    with os.fdopen(read, 'rb') as incoming:
        serve(path, pin, 1, incoming, output, factory=Controller)
    assert events == ['started', 'closed']
    assert [json.loads(line)['op'] for line in output.getvalue().splitlines()] == ['ready', 'closed']


def test_bootstrap_refuses_changed_or_public_profile_before_dispatch(tmp_path):
    path, pin = profile(tmp_path)
    with pytest.raises(ValueError, match='BOOTSTRAP_PIN'):
        load_profile(path, '0' * 64)
    path.chmod(0o644)
    with pytest.raises(ValueError, match='BOOTSTRAP_PROFILE'):
        load_profile(path, pin)
