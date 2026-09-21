"""Capture versioned foreign snapshots only when preparing a new native turn."""
from types import SimpleNamespace
import pytest
from test_glm_restore import Prefix, Publication, Http, body
from test_glm_snapshot_bank import make_bank, update
from drift.serving.glm_restore_turn import TurnRestorer, RestoringTransport


def bound_session(tmp_path, monkeypatch):
    from drift.serving.glm_snapshot_binding import bind_snapshot_bank
    import drift.serving.glm_snapshot_binding as module
    bank, root = make_bank(tmp_path); first = update(root, 1); bank.publish(first)
    events, copies = [], []
    transport_root = tmp_path / 'transport'; transport_root.mkdir()
    fixed = TurnRestorer(Prefix(), None, rows=2, digest=first['sha256'], root=transport_root,
                         deadline=lambda: 9999999999, max_input_bytes=4096)
    inner = Http(events)
    session = SimpleNamespace(started=None, poisoned=False, restoration=fixed, transport=RestoringTransport(inner, fixed),
                              restoration_ownership={'source_worker':'qwen', 'target_worker':'glm'}, limits={'max_input_bytes':4096})
    def count(value, timeout):
        inner.bodies.append(value); return 10 + session.restoration.rows
    inner.count_tokens = count
    class Route:
        spec = {'peer': 'fixture'}
        def validate(self): pass
        def bind(self, value): return value
    def publication(path, digest, name, peer, check):
        copies.append((path, digest)); return Publication(events)
    monkeypatch.setattr(module, 'PublicationTransport', publication)
    bind_snapshot_bank(session, bank, Route(), translator_sha256='a'*64)
    return session, bank, root, copies


def test_pending_update_cannot_change_prepared_request_or_receipts(tmp_path, monkeypatch):
    session, bank, root, copies = bound_session(tmp_path, monkeypatch)
    request = {**body(), **session.prepare_turn(body())}
    first = bank.capture(); bank.publish(update(root, 2, rows=3))
    session.transport.count_tokens(request, 5); list(session.transport.stream(request, 5))
    assert copies[0][1] == first.sha256
    assert session.restoration.report()['turns'][0]['snapshot']['version'] == 1
    request = {**body(), **session.prepare_turn(body())}
    session.transport.count_tokens(request, 5); list(session.transport.stream(request, 5))
    turns = session.restoration.report()['turns']
    assert turns[1]['snapshot']['version'] == 2 and copies[1][1] != copies[0][1]
    assert turns[1]['snapshot']['rows'] == 3 and turns[1]['reserve_tokens'] == 3
    assert [turn['snapshot_sha256'] for turn in turns] == [value[1] for value in copies]
    assert all(turn['snapshot']['source_worker'] == 'qwen' for turn in turns)


def test_second_prepare_never_changes_active_version(tmp_path, monkeypatch):
    session, bank, root, _ = bound_session(tmp_path, monkeypatch)
    session.prepare_turn(body()); bank.publish(update(root, 2, rows=3))
    with pytest.raises(ValueError): session.prepare_turn(body())
    assert session.restoration.rows == 2


def test_binding_requires_unopened_session_and_matching_recipe(tmp_path, monkeypatch):
    from drift.serving.glm_snapshot_binding import bind_snapshot_bank
    session, bank, _, _ = bound_session(tmp_path, monkeypatch)
    with pytest.raises(ValueError): bind_snapshot_bank(session, bank, None, translator_sha256='b'*64)
    session.started = 10
    with pytest.raises(ValueError): bind_snapshot_bank(session, bank, None, translator_sha256='a'*64)


def test_update_during_prefix_verification_is_only_used_next_turn(tmp_path, monkeypatch):
    session, bank, root, copies = bound_session(tmp_path, monkeypatch)
    first = bank.capture()
    class UpdatingPrefix(Prefix):
        def verify(self, original, reserved, rows, timeout):
            bank.publish(update(root, 2, rows=3))
            return super().verify(original, reserved, rows, timeout)
    session.restoration.verifier = UpdatingPrefix()
    session.prepare_turn(body())
    assert bank.capture().version == 2
    assert copies[0][1] == first.sha256 and session.restoration.rows == 2


def test_bank_corruption_after_preparation_suppresses_reader_events(tmp_path, monkeypatch):
    session, bank, root, _ = bound_session(tmp_path, monkeypatch)
    request = {**body(), **session.prepare_turn(body())}
    session.transport.count_tokens(request, 5)
    frame = update(root, 2); frame['source_worker'] = 'wrong'
    with pytest.raises(ValueError): bank.publish(frame)
    with pytest.raises(ValueError): next(session.transport.stream(request, 5))
    assert session.transport.inner.closed
    assert 'dispatch' not in session.transport.inner.events
