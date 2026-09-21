"""Require completed native turns, explicit release or invalidation, and owned cleanup."""


def assess(report):
    parked = report['parked']
    pair = bool(parked) and len(report['workers']) == 2 and parked['workers'].keys() == report['workers'].keys()
    complete = pair and all(worker.get('terminal', 0) > 0 and worker.get('terminal') == worker.get('stream') for worker in parked['workers'].values())
    calls = [event for event in report['events'] if event['kind'] == 'tool_call']
    ordered = bool(calls) and all(event['terminals'] == event['streams'] for event in calls)
    admitted = [event for event in report['events'] if event['kind'] == 'admitted' and event['tool'] == 'hub']
    closed = pair and all(worker.get('close') == 1 for worker in report['workers'].values())
    cleanup = report['cleanup']
    clean = all(cleanup.get(key) is True for key in ('reaped', 'joined', 'group_gone', 'worker_pids_gone')) and not cleanup.get('terminated') and not cleanup.get('output_limit')
    if report['mode'] in ('release', 'timeout'):
        released = report['released_at']
        safe = released is not None and bool(admitted) and all(event['at'] >= released for event in admitted) and not report['failed_roles'] and report['exit_code'] == 0
    else:
        safe = not admitted and not report['ready_remaining'] and set(report['failed_roles']) == {'parent', 'child'}
        if report['mode'] == 'abort': safe = safe and report['abort_sent'] and report['abort_ack']
    boundary_ok = bool(safe and complete and ordered)
    verdict = 'PASSED' if boundary_ok and clean and closed else 'BLOCKED' if boundary_ok and clean or report['mode'] == 'abort' and not pair else 'FAILED'
    return dict(verdict=verdict, boundary_verdict='PASSED' if boundary_ok else 'BLOCKED' if report['mode'] == 'abort' and not pair else 'FAILED', process_cleanup_verdict='PASSED' if clean else 'FAILED', graceful_close_verdict='PASSED' if closed else 'BLOCKED', terminal_before_hooks=ordered, both_ready=pair, generation_complete_at_pair=complete, both_workers_closed=closed, clean_process_exit=clean, hub_admissions=len(admitted))
