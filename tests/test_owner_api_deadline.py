import inspect
import pytest
import test_glm_owner_worker as glm
import test_worker_owner_socket as qwen
from drift.serving.glm_owner_worker import serve_owner
from drift.serving.worker_owner_stdio import launch


@pytest.fixture
def private_root(tmp_path):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix='apiown-', dir='/tmp') as value:
        root = Path(value).resolve()
        root.chmod(0o700)
        yield root


def test_qwen_explicit_180_second_owner_reaps_on_prompt_eof(private_root):
    child, root, _ = qwen.launch(private_root, wall=180)
    try:
        assert qwen.line(child)['ready']
        with qwen.connect(root):
            child.stdin.close()
            receipt = qwen.receipt(child, root)
        assert receipt['status'] == 'PASSED' and receipt['reason'] == 'input_eof'
        assert receipt['wall_seconds'] < 5
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_qwen_above_180_rejected_before_child_or_evidence(private_root):
    child, root, _ = qwen.launch(private_root, wall=181)
    stdout, stderr = child.communicate(timeout=3)
    assert child.returncode == 2 and stdout == stderr == b''
    assert not (root/'termination.json').exists()
    assert not list(root.glob('worker-owner-*'))


def test_qwen_default_owner_wall_remains_60():
    assert inspect.signature(launch).parameters['wall_seconds'].default == 60


@pytest.mark.parametrize('milliseconds', [170000, 180000])
def test_glm_explicit_api_deadline_closes_under_actual_supervisor(private_root, monkeypatch, milliseconds):
    original = glm.configuration
    def config(root):
        value = original(root)
        value['limits']['deadline_ms'] = milliseconds
        return value
    monkeypatch.setattr(glm, 'configuration', config)
    process, configuration = glm.start(private_root)
    try:
        with glm.opened(process, configuration):
            glm.send(process, 3, 'close', {})
            assert glm.read(process)['op'] == 'closed'
            report = glm.finish(process, private_root)
        assert report['status'] == 'PASSED'
        assert report['wall_seconds'] < 5
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


@pytest.mark.parametrize('milliseconds', [180001, True, 170000.0])
def test_glm_invalid_deadline_refuses_before_backend(milliseconds):
    calls = []
    with pytest.raises(ValueError, match='OWNER_LIMIT'):
        serve_owner({'limits': {'deadline_ms': milliseconds}}, backend_factory=lambda cfg: calls.append(cfg))
    assert calls == []


def test_qwen_extended_budget_owner_abort_still_reaps(private_root):
    child, root, _ = qwen.launch(private_root, wall=180)
    try:
        assert qwen.line(child)['ready']
        with qwen.connect(root) as peer:
            peer.sendall(b'{"op":"abort"}\n')
            report = qwen.receipt(child, root)
        assert report['status'] == 'FAILED' and report['reason'] == 'owner_control'
        assert report['wall_seconds'] < 5
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_glm_extended_budget_owner_eof_poisoned_and_reaped(private_root, monkeypatch):
    original = glm.configuration
    def config(root):
        value = original(root)
        value['limits']['deadline_ms'] = 170000
        return value
    monkeypatch.setattr(glm, 'configuration', config)
    process, configuration = glm.start(private_root)
    try:
        peer = glm.opened(process, configuration)
        peer.close()
        report = glm.finish(process, private_root)
        assert report['status'] == 'FAILED' and report['bank_poisoned'] and report['control_failed']
        assert report['wall_seconds'] < 5
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
