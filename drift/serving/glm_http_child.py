"""Own one localhost HTTP request and emit bounded private protocol frames."""
import json
import os
import sys
import urllib.request
from urllib.parse import urlsplit


def main():
    raw = sys.stdin.buffer.readline(1048577)
    if len(raw) > 1048576:
        raise ValueError('input limit')
    request = json.loads(raw)
    url = urlsplit(request['url'])
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password or url.query or url.fragment:
        raise ValueError('loopback required')
    if url.path not in ('/tokenize', '/v1/chat/completions'):
        raise ValueError('unsupported endpoint')
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('DRIFT_GLM_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    body = json.dumps(request['body'], allow_nan=False).encode()
    operation = urllib.request.Request(request['url'], data=body, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(operation, timeout=request['timeout']) as response:
        if url.path == '/tokenize':
            encoded = response.read(2097153)
            if len(encoded) > 2097152:
                raise ValueError('tokenization limit')
            tokens = json.loads(encoded)['tokens']
            if type(tokens) is not list or any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError('tokenization protocol')
            print(json.dumps({'tokens': len(tokens)}), flush=True)
            return
        total = 0
        while True:
            line = response.readline(131073)
            if not line:
                raise ValueError('missing stream terminator')
            total += len(line)
            if len(line) > 131072 or total > 4194304:
                raise ValueError('stream bound')
            line = line.strip()
            if not line or line.startswith(b':'):
                continue
            if line == b'data: [DONE]':
                print('{"done":true}', flush=True)
                return
            if not line.startswith(b'data: '):
                raise ValueError('invalid SSE frame')
            print(json.dumps({'event': json.loads(line[6:])}), flush=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('redirect forbidden')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('{"transport_error":"HTTP_REQUEST_FAILED"}', flush=True)
        raise SystemExit(2)
