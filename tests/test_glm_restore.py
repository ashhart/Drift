import copy
import time
import pytest
from drift.serving.glm_restore_prompt import reserve_prompt, verify_prefix
from drift.serving.glm_restore_turn import TurnRestorer, RestoringTransport


def body():
    return {'model': 'fixture', 'messages': [{'role': 'system', 'content': 'own instructions'}, {'role': 'user', 'content': 'own task'}], 'tools': [{'type': 'function', 'function': {'name': 'inspect'}}], 'tool_choice': 'auto', 'max_tokens': 8, 'stream': True}


def test_reservation_preserves_every_own_field_and_never_uses_template_override():
    original = body(); before = copy.deepcopy(original)
    altered = reserve_prompt(original, 3)
    assert original == before and 'chat_template' not in altered
    assert altered['messages'][0]['content'] == '[MASK]' * 3 + original['messages'][0]['content']
    altered['messages'][0]['content'] = original['messages'][0]['content']
    assert altered == original


def test_prefix_verification_rejects_boundary_retokenization_and_marker_in_own_prompt():
    assert verify_prefix([10, 11, 12], [10, 154821, 154821, 11, 12], 2)['reserve_start'] == 1
    with pytest.raises(ValueError): verify_prefix([10, 11, 12], [10, 154821, 154821, 13, 12], 2)
    with pytest.raises(ValueError): verify_prefix([154821, 11], [154821] * 3 + [11], 2)


class Prefix:
    def verify(self, original, reserved, rows, timeout):
        assert reserved == reserve_prompt(original, rows)
        return {'verified': True, 'own_tokens': 10, 'prompt_tokens': 10 + rows, 'reserve_start': 3, 'reserve_tokens': rows}
    def close(self): pass


class Delivery:
    def __init__(self, events, fail=False): self.events, self.fail = events, fail
    def health(self, **kwargs): self.events.append('health')
    def admit(self, *args, **kwargs): self.events.append('admit')
    def wait_applied(self, sequence, rows, digest, **kwargs):
        self.events.append('receipts')
        if self.fail: raise ValueError('rank failure')
        return [{'rank': rank, 'world_size': 2, 'sequence': sequence, 'rows': rows, 'sha256': digest} for rank in range(2)]


class Publication:
    def __init__(self, events, fail=False): self.events, self.delivery = events, Delivery(events, fail)
    def stage(self, **kwargs): self.events.append('stage')
    def publish(self, **kwargs): self.events.append('publish')


class Http:
    def __init__(self, events): self.events, self.bodies, self.closed = events, [], False
    def count_tokens(self, value, timeout):
        self.bodies.append(copy.deepcopy(value)); return 12
    def stream(self, value, timeout):
        self.events.append('dispatch'); self.bodies.append(copy.deepcopy(value))
        yield {'choices': [{'delta': {'content': 'private generated text'}}]}
    def close(self): self.closed = True
    def cancel(self): self.close()


def setup(tmp_path, linked=True, fail=False):
    events, names = [], []
    def publication(name): names.append(name); return Publication(events, fail)
    restorer = TurnRestorer(Prefix(), publication, rows=2, digest='a' * 64, root=tmp_path,
                           deadline=lambda: time.monotonic() + 5, max_input_bytes=4096, linked=linked)
    transport = RestoringTransport(Http(events), restorer)
    return restorer, transport, events, names


def test_each_turn_is_fresh_and_rank_receipts_precede_releasing_native_events(tmp_path):
    restorer, transport, events, names = setup(tmp_path)
    for _ in range(2):
        original = body(); extra = restorer.prepare(copy.deepcopy(original)); request = {**original, **extra}
        assert transport.count_tokens(request, 5) == 12
        stream = transport.stream(request, 5)
        assert next(stream)['choices']
        assert events[-2:] == ['dispatch', 'receipts']
        list(stream)
    assert len(set(names)) == 2 and all((tmp_path / 'tp-live-in' / name).is_dir() for name in names)
    assert all('chat_template' not in value for value in transport.inner.bodies)
    assert restorer.report()['completed_turns'] == 2


def test_missing_rank_receipt_never_releases_native_tool_or_text(tmp_path):
    restorer, transport, events, _ = setup(tmp_path, fail=True)
    request = {**body(), **restorer.prepare(body())}
    transport.count_tokens(request, 5)
    with pytest.raises(ValueError): next(transport.stream(request, 5))
    assert transport.inner.closed


def test_no_link_uses_identical_reservation_without_publication(tmp_path):
    restorer, transport, events, names = setup(tmp_path, linked=False)
    request = {**body(), **restorer.prepare(body())}
    assert 'kv_transfer_params' not in request
    transport.count_tokens(request, 5); list(transport.stream(request, 5))
    assert names == [] and events == ['dispatch']
    assert transport.inner.bodies[0]['messages'] == reserve_prompt(body(), 2)['messages']


def test_changed_own_messages_after_preparation_cannot_dispatch(tmp_path):
    restorer, transport, _, _ = setup(tmp_path)
    request = {**body(), **restorer.prepare(body())}; request['messages'][1]['content'] = 'changed'
    with pytest.raises(ValueError): transport.count_tokens(request, 5)
    assert transport.inner.bodies == []


def test_session_deadline_does_not_refresh_between_staging_operations(tmp_path):
    now, deadlines = [10.0], []
    class TimedPublication(Publication):
        def stage(self, *, timeout):
            deadlines.append(timeout); now[0] = 12.0
    restorer = TurnRestorer(Prefix(), lambda _: TimedPublication([]), rows=2, digest='a' * 64,
                           root=tmp_path, deadline=lambda: 11.0, clock=lambda: now[0], max_input_bytes=4096)
    with pytest.raises(TimeoutError): restorer.prepare(body())
    assert deadlines == [1.0] and restorer.plan is None


def test_reservation_extras_cannot_escape_input_byte_cap(tmp_path):
    restorer, transport, _, _ = setup(tmp_path, linked=False)
    restorer.max_input_bytes = len(__import__('json').dumps(reserve_prompt(body(), 2)).encode()) + 1
    with pytest.raises(ValueError): restorer.prepare(body())
    assert transport.inner.bodies == []


def test_memory_parent_symlink_is_rejected_before_publication(tmp_path):
    restorer, _, events, _ = setup(tmp_path)
    outside = tmp_path / 'outside'; outside.mkdir()
    (tmp_path / 'tp-live-in').symlink_to(outside)
    with pytest.raises(ValueError): restorer.prepare(body())
    assert events == []


def test_taps_stay_off_by_default_and_opt_in_exports_only_through_hook(tmp_path):
    class Outbox:
        def __init__(self): self.events=[]
        def begin(self,name,proof,maximum): self.events.append(('begin',name,proof['prompt_tokens'],maximum))
        def finish(self,name,maximum,deadline):
            self.events.append(('finish',name,maximum));return {'full_completion':False,'final_tail':'UNKNOWN'}
        def close(self): pass
    restorer, transport, _, _ = setup(tmp_path)
    outbox=Outbox();restorer.outbox=outbox
    request={**body(),**restorer.prepare(body())}
    assert request['kv_transfer_params']['drift_tap'] is True
    assert outbox.events[0][0]=='begin'
    transport.count_tokens(request,5);request['max_tokens']=3
    list(transport.stream(request,5))
    assert outbox.events[-1][2]==3
    assert restorer.report()['turns'][0]['outbox']['full_completion'] is False
    other=tmp_path/'other';other.mkdir()
    restorer,_,_,_=setup(other)
    assert restorer.prepare(body())['kv_transfer_params']['drift_tap'] is False
