"""Drive one linked GLM publication and observe scoped cancellation recovery."""
import time

from drift.serving.live_single_publication import SinglePublication


def drive_linked(observer, client, sample, transport, *, rows, digest, max_seconds=180,
                 poll_seconds=.25, pause=time.sleep, clock=time.monotonic):
    began, launched, aborted = clock(), False, False
    publication = None
    result = {'status': 'INVALID', 'reason': 'qualification_did_not_complete'}
    try:
        while True:
            if clock() - began >= max_seconds: raise TimeoutError('linked qualification deadline exceeded')
            if launched: client.poll(timeout=0)
            if launched and client.publication_ready and publication is None:
                gate = SinglePublication(transport.delivery, rows=rows, digest=digest, timeout=observer.remaining())
                def release(*, timeout):
                    started = clock()
                    transport.publish(timeout=timeout)
                    if clock() - started >= timeout: raise TimeoutError('publication release deadline exceeded')
                    client.release()
                publication = gate.run(transport.stage, release)
                if publication['status'] != 'PASSED': raise RuntimeError('publication was not verified')
            result = observer.poll(sample, client.request_state)
            if result['status'] in {'PASSED', 'FAILED', 'INVALID', 'BLOCKED'}: break
            if result.get('action') == 'start':
                if launched: raise RuntimeError('duplicate linked start')
                client.start()
                launched = True
            elif result.get('action') == 'abort':
                if aborted or publication is None or publication['status'] != 'PASSED':
                    raise RuntimeError('abort lacked publication evidence')
                client.abort()
                aborted = True
            pause(min(poll_seconds, observer.remaining()))
    except Exception as error:
        result = dict(result, status='INVALID', action=None, reason='linked_driver_error', driver_error=type(error).__name__)
    finally:
        try: client.close()
        except Exception as error:
            result = dict(result, status='INVALID', action=None, reason='cleanup_error', cleanup_error=type(error).__name__)
    try: client_report = client.report()
    except Exception as error:
        client_report = {}
        result = dict(result, status='INVALID', reason='client_report_error', client_report_error=type(error).__name__)
    result.update(client=client_report, publication=publication, total_seconds=clock() - began,
                  scope='single_linked_glm_publication_and_cancellation', qwen_shutdown='BLOCKED',
                  causal_use='NOT_TESTED', atomic_tp='NOT_ESTABLISHED')
    if result['status'] == 'PASSED' and (not launched or not aborted or publication is None
                                       or publication['status'] != 'PASSED' or result['total_seconds'] >= max_seconds):
        result.update(status='INVALID', reason='incomplete_or_late_linked_evidence')
    return result
