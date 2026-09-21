"""Represent explicit own-control updates as new private input without rewriting history."""
from drift.serving.worker_contract import fields, require


def own_control_message(payload, *, additive=False):
    fields(payload, ('system_prompt',))
    values = payload['system_prompt']
    require(type(values) is list and all(type(value) is str for value in values))
    require(not additive or bool(values))
    content = '\n\n'.join(values) if values else '(No additional own control instructions.)'
    prefix = ('Additional own control context; earlier instructions remain in effect:\n' if additive else
              'Current own control snapshot; supersedes earlier own control snapshots and additions only:\n')
    return {'role': 'system', 'content': prefix + content}


def apply_own_control(session, payload, *, additive=False):
    own_control_message(payload, additive=additive)
    operation = 'own_control_append' if additive else 'own_control'
    require(session.allow_own_control and callable(getattr(session.backend, operation, None)), 'CAPABILITY')
    require(not session.ready)
    require(session.control_updates < session.limits['max_turns'], 'LIMIT')
    getattr(session.backend, operation)(payload)
    session._deadline()
    session.control_updates += 1
    session.control_ready = True
    return operation + '_ack', {}
