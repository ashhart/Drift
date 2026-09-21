"""Drive one owned cancellation attempt and return aggregate evidence only."""
import time


def drive(observer, client, sample, *, poll_seconds=0.25, max_seconds=180,
          pause=time.sleep, clock=time.monotonic):
    started_at, launched, aborted = clock(), False, False
    result = {'status': 'INVALID', 'reason': 'qualification_did_not_complete'}
    try:
        while True:
            if clock() - started_at >= max_seconds:
                raise TimeoutError('qualification deadline reached')
            if launched:
                client.poll(timeout=0)
            result = observer.poll(sample, client.request_state)
            if result['status'] in {'PASSED', 'FAILED', 'INVALID', 'BLOCKED'}:
                break
            if result.get('action') == 'start':
                if launched:
                    raise RuntimeError('duplicate start action')
                client.start()
                launched = True
            elif result.get('action') == 'abort':
                if not launched or aborted:
                    raise RuntimeError('invalid abort action')
                client.abort()
                aborted = True
            pause(min(poll_seconds, max(0, max_seconds - (clock() - started_at))))
    except Exception as error:
        result = dict(result, status='INVALID', action=None, reason='driver_error', driver_error=type(error).__name__)
    finally:
        try:
            client.close()
        except Exception as error:
            result = dict(result, status='INVALID', action=None, reason='cleanup_error', cleanup_error=type(error).__name__)
    try:
        client_report = client.report()
    except Exception as error:
        client_report = {}
        result = dict(result, status='INVALID', action=None, reason='client_report_error', client_report_error=type(error).__name__)
    result = dict(result, client=client_report, total_seconds=clock() - started_at)
    if result['status'] == 'PASSED' and (not launched or not aborted or result['total_seconds'] > max_seconds):
        result.update(status='INVALID', reason='incomplete_or_late_driver_evidence')
    return result
