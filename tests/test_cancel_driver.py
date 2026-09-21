"""Check qualification orchestration with synthetic observer and client objects."""
from types import SimpleNamespace as NS

from scripts.qualify_live.cancel_driver import drive


class Client:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append('start')

    def poll(self, timeout=0):
        self.calls.append('poll')

    def request_state(self):
        return NS()

    def abort(self):
        self.calls.append('abort')

    def close(self):
        self.calls.append('close')

    def report(self):
        return {'closed': bool(self.calls) and self.calls[-1] == 'close'}


class Observer:
    def __init__(self, actions):
        self.actions = iter(actions)

    def poll(self, sample, state):
        value = next(self.actions)
        if isinstance(value, Exception):
            raise value
        return value


def test_driver_closes_owned_client_after_verified_cancel():
    client = Client()
    observer = Observer([{'status': 'RUNNING', 'action': 'start'},
                         {'status': 'RUNNING', 'action': 'abort'},
                         {'status': 'PASSED', 'action': None}])
    result = drive(observer, client, lambda **kw: None, pause=lambda delay: None)
    assert client.calls == ['start', 'poll', 'abort', 'poll', 'close']
    assert result['status'] == 'PASSED' and result['client']['closed']


def test_driver_never_starts_on_failed_idle_preflight():
    client = Client()
    result = drive(Observer([{'status': 'BLOCKED', 'action': None}]), client, lambda **kw: None)
    assert client.calls == ['close'] and result['status'] == 'BLOCKED'


def test_driver_preserves_failure_and_discards_exception_payload():
    client = Client()
    observer = Observer([{'status': 'RUNNING', 'action': 'start', 'baseline': {'running': 0}}, RuntimeError('private-payload')])
    result = drive(observer, client, lambda **kw: None, pause=lambda delay: None)
    assert client.calls[-1] == 'close'
    assert result['status'] == 'INVALID' and 'private-payload' not in str(result)
    assert result['baseline'] == {'running': 0}


def test_driver_rejects_pass_when_cleanup_fails():
    client = Client()
    def bad_close():
        raise RuntimeError('private-payload')
    client.close = bad_close
    result = drive(Observer([{'status': 'PASSED', 'action': None}]), client, lambda **kw: None)
    assert result['status'] == 'INVALID' and 'private-payload' not in str(result)


def test_driver_records_client_report_failure_without_losing_the_evidence():
    client = Client()
    def bad_report():
        raise RuntimeError('private-payload')
    client.report = bad_report
    observer = Observer([{'status': 'FAILED', 'action': None, 'samples': 3}])
    result = drive(observer, client, lambda **kw: None)
    assert result['status'] == 'INVALID' and result['samples'] == 3
    assert result['client'] == {} and result['client_report_error'] == 'RuntimeError'
    assert 'private-payload' not in str(result)
