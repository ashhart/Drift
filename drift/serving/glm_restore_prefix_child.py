"""Verify native placeholder tokenization privately and emit only numeric evidence."""
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_http_child import NoRedirect
from glm_restore_prompt import reserve_prompt, verify_prefix


def main():
    raw = sys.stdin.buffer.readline(1048577)
    if len(raw) > 1048576: raise ValueError('input bound')
    request = json.loads(raw)
    url = urlsplit(request['url'])
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password or url.path != '/verify-reservation' or url.query or url.fragment:
        raise ValueError('private loopback reservation operation required')
    body = request['body']; rows = body['rows']
    original, reserved = body['original'], body['reserved']
    if reserved != reserve_prompt(original, rows): raise ValueError('own input changed')
    deadline = time.monotonic() + request['timeout']
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('DRIFT_GLM_KEY'): headers['Authorization'] = 'Bearer ' + os.environ['DRIFT_GLM_KEY']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def tokenize(value):
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError('reservation deadline')
        keys = ('model', 'messages', 'tools', 'chat_template_kwargs')
        data = {key: value[key] for key in keys if key in value}
        data['add_generation_prompt'] = True
        operation = urllib.request.Request(url._replace(path='/tokenize').geturl(), data=json.dumps(data).encode(), headers=headers)
        with opener.open(operation, timeout=remaining) as response:
            encoded = response.read(2097153)
            if len(encoded) > 2097152: raise ValueError('tokenizer response bound')
            result = json.loads(encoded)['tokens']
        if time.monotonic() >= deadline: raise TimeoutError('reservation deadline')
        return result
    proof = verify_prefix(tokenize(original), tokenize(reserved), rows)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception:
        print('{"transport_error":"PREFIX_VERIFICATION_FAILED"}', flush=True)
        raise SystemExit(2)
