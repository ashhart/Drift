"""Exercise the private native-tokenizer proof through its real owned child."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from drift.serving.glm_restore_prefix import PrefixVerifier
from drift.serving.glm_restore_prompt import reserve_prompt


@pytest.mark.parametrize('corrupt', [False, True])
def test_real_prefix_child_never_exports_ids_and_reaps(corrupt):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            tokens = [8, 9, 10]
            if body['messages'][0]['content'].startswith('[MASK]'):
                tokens = [8, 154821, 154821, 9, 11 if corrupt else 10]
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'tokens': tokens}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    verifier = PrefixVerifier(f'http://127.0.0.1:{server.server_port}')
    body = {'model': 'fixture', 'messages': [{'role': 'system', 'content': 'own'}], 'tools': []}
    try:
        if corrupt:
            with pytest.raises(ValueError): verifier.verify(body, reserve_prompt(body, 2), 2, 3)
        else:
            assert verifier.verify(body, reserve_prompt(body, 2), 2, 3) == {
                'verified': True, 'own_tokens': 3, 'prompt_tokens': 5, 'reserve_start': 1, 'reserve_tokens': 2}
        assert len(seen) == 2 and verifier.child is None
        assert all(request['add_generation_prompt'] is True for request in seen)
    finally:
        verifier.close(); server.shutdown(); server.server_close(); thread.join(2)
