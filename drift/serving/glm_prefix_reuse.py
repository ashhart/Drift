"""Client-side policy for prefix-cache reuse under a linked GLM session; rationale in docs/GLM_PREFIX_REUSE.md."""
import copy
import hashlib
import re

from drift.serving.glm_restore_prompt import MARKER

ALIGN = 3584
_SALT = re.compile(r'drift:[0-9a-f]{32}')
_NAME = re.compile(r'[A-Za-z0-9_.-]{1,96}')


def session_salt(session_secret: str) -> str:
    """One salt per worker session; the caller must supply a fresh owner-held random secret."""
    if type(session_secret) is not str or len(session_secret) < 16:
        raise ValueError('session secret too short for a cache salt')
    return 'drift:' + hashlib.sha256(('cache-salt:' + session_secret).encode()).hexdigest()[:32]


def _salt(salt):
    if type(salt) is not str or not _SALT.fullmatch(salt):
        raise ValueError('cache salt must come from session_salt')
    return salt


def _own_system(body):
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages or messages[0].get('role') != 'system' or type(messages[0].get('content')) is not str:
        raise ValueError('requires an existing own system string')
    if MARKER in messages[0]['content']:
        raise ValueError('own system text already contains the marker')
    return messages[0]


def reserve_after_system(body, rows):
    """Append the marker run to the END of the own system text, so the stable text precedes the span."""
    if type(rows) is not int or not 1 <= rows <= 4096 or 'chat_template' in body:
        raise ValueError('unsupported reserved native prompt')
    system = _own_system(body)
    result = copy.deepcopy(body)
    result['messages'][0]['content'] = system['content'] + MARKER * rows
    return result


def warm_request(body, salt):
    """No-link, one-token request that lets the engine store the ORIGINAL (unreserved) stable prefix."""
    system = _own_system(body)
    warm = {key: copy.deepcopy(body[key]) for key in ('model', 'tools', 'tool_choice', 'chat_template_kwargs') if key in body}
    warm.update(messages=[copy.deepcopy(system), {'role': 'user', 'content': 'ok'}], max_tokens=1, temperature=0, stream=False, cache_salt=_salt(salt))
    return warm


def linked_extra(salt, kv_transfer_params):
    """Extra body fields for a linked live turn that may read, but never write, the prefix cache."""
    session, reserve, start = (kv_transfer_params.get(key) for key in ('drift_session', 'drift_reserve', 'drift_reserve_start'))
    if type(session) is not str or not _NAME.fullmatch(session):
        raise ValueError('prefix reuse applies to named live sessions only')
    if type(reserve) is not int or not 1 <= reserve <= 4096 or type(start) is not int or start < 0:
        raise ValueError('invalid reserved span')
    return {'cache_salt': _salt(salt), 'vllm_xargs': {'skip_writing_prefix_cache': 1}, 'kv_transfer_params': {**kv_transfer_params, 'drift_prefix_reuse': True}}


def reusable_tokens(warm_tokens, linked_tokens, reserve_start, align=ALIGN):
    """Upper bound on cache-served tokens: common prefix, cut at the span, rounded DOWN to the alignment unit."""
    if type(align) is not int or align <= 0 or type(reserve_start) is not int or reserve_start < 0:
        raise ValueError('alignment must be a positive integer and the span start non-negative')
    if any(type(t) is not int for tokens in (warm_tokens, linked_tokens) for t in tokens):
        raise ValueError('token ids must be integers')
    shared = next((i for i, pair in enumerate(zip(warm_tokens, linked_tokens)) if pair[0] != pair[1]), min(len(warm_tokens), len(linked_tokens)))
    return min(shared, reserve_start) // align * align
