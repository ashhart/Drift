"""Reserve only local placeholder input and verify its exact native token insertion."""
import copy

MARKER, MARKER_ID = '[MASK]', 154821


def reserve_prompt(body, rows):
    if type(rows) is not int or not 1 <= rows <= 4096 or 'chat_template' in body:
        raise ValueError('unsupported reserved native prompt')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages or messages[0].get('role') != 'system' or type(messages[0].get('content')) is not str:
        raise ValueError('reservation requires an existing own system string')
    result = copy.deepcopy(body)
    result['messages'][0]['content'] = MARKER * rows + messages[0]['content']
    return result


def verify_prefix(original, reserved, rows):
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError('invalid reservation bound')
    for tokens in (original, reserved):
        if type(tokens) is not list or not 0 < len(tokens) <= 65536 or any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError('invalid private tokenizer response')
    if MARKER_ID in original or len(reserved) != len(original) + rows:
        raise ValueError('ambiguous reservation or changed native token count')
    start = next((index for index, pair in enumerate(zip(original, reserved)) if pair[0] != pair[1]), len(original))
    if reserved[start:start + rows] != [MARKER_ID] * rows or reserved[start + rows:] != original[start:] or start >= len(original):
        raise ValueError('native token prefix or suffix changed')
    return {'verified': True, 'own_tokens': len(original), 'prompt_tokens': len(reserved),
            'reserve_start': start, 'reserve_tokens': rows}
