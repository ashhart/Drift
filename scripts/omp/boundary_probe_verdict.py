"""Assess public boundary observations without treating CLI exit as gate success."""


def assess(report):
    calls = [event for event in report['events'] if event['kind'] == 'tool_call']
    parked = report['parked']
    workers = parked['workers'].values() if parked else []
    pair = bool(parked) and len(report['workers']) == 2 and parked['workers'].keys() == report['workers'].keys()
    generated = bool(calls) and all(event['terminals'] == event['streams'] for event in calls)
    still_parked = pair and all(worker['terminal'] == worker['stream'] for worker in workers)
    hub_admissions = [event for event in report['events'] if event['kind'] == 'admitted' and event['tool'] == 'hub']
    closed = pair and all(worker.get('close') == 1 for worker in report['workers'].values())
    if report['mode'] == 'release':
        released = report['released_at']
        safe = generated and still_parked and released is not None and bool(hub_admissions) and all(event['at'] >= released for event in hub_admissions) and closed and report['exit_code'] == 0
    else:
        safe = generated and pair and closed and not hub_admissions and not report['ready_remaining']
    return dict(verdict='PASSED' if safe else 'FAILED', terminal_before_hooks=generated, both_ready=bool(parked), generation_still_complete_at_pair=still_parked, both_workers_closed=closed, hub_admissions=len(hub_admissions), one_bun_process=len({event['pid'] for event in report['events']}) == 1)
