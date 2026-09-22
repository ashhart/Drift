"""Exercise owner CLI pinning without loading a model or native library."""
import hashlib
import json

import pytest


def test_cli_carries_both_pinned_configs(tmp_path, monkeypatch):
    from drift.serving import mcdma_glm_owner as cli
    calls, arguments = [], []
    for name in ('config', 'runtime'):
        raw = json.dumps({'fixture': name}).encode()
        path = tmp_path / f'{name}.json'
        path.write_bytes(raw)
        arguments += ['--' + name, str(path), '--' + name + '-sha256', hashlib.sha256(raw).hexdigest()]
    monkeypatch.setattr(cli, 'launch', lambda *args: calls.append(args) or 0)
    assert cli.main(arguments) == 0
    assert calls == [({'fixture': 'config'}, {'fixture': 'runtime'})]
    arguments[-1] = '0' * 64
    assert cli.main(arguments) == 2 and len(calls) == 1


@pytest.mark.parametrize('runtime', [{}, {'v': True}, {'v': 2}])
def test_cli_refuses_incomplete_runtime_before_code_loading(runtime, monkeypatch):
    from drift.serving import mcdma_glm_owner as cli
    monkeypatch.setattr(cli, 'load_opener', lambda **kw: pytest.fail('native code loaded'))
    with pytest.raises(ValueError): cli.launch({}, runtime)
