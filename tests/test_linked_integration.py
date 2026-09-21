"""Run real linked client, runner and transport against isolated local HTTP/rank fixtures."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from drift.serving.live_cancel_observation import CancellationObserver
from drift.serving.live_linked_client import LinkedSession
from drift.serving.live_metric_fetch import fetch_metrics
from drift.serving.live_metrics import parse_metrics
from drift.serving.live_publication_transport import PublicationTransport
from drift.serving.live_qualification_client import Limits, claim_session
from scripts.qualify_live.linked_driver import drive_linked

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('fault', [None, 'missing_peer_receipt', 'natural_completion'])
def test_real_linked_driver_retains_receipts_and_stops_owned_stream(tmp_path, monkeypatch, fault):
    active, disconnected = threading.Event(), threading.Event()
    rank_roots = [tmp_path / 'head', tmp_path / 'peer']
    requests = []
    claim = claim_session(tmp_path / 'ledger')
    source = tmp_path / 'publication.npz'
    np.savez(source, **{f'l{3+4*i}': np.ones((12, 512), np.float32) for i in range(11)})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def do_GET(self):
            count = int(active.is_set())
            values = {'num_requests_running': count, 'num_requests_waiting': 0, 'kv_cache_usage_perc': count/10}
            body = '\n'.join(f'vllm:{name}{{engine="0",model_name="GLM-5.3-Flash-EXL3"}} {value}' for name,value in values.items()).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                payload = b'{"tokens":[42]}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            requests.append(body)
            assert body['kv_transfer_params']['drift_tap'] is False
            for rank, folder in enumerate(rank_roots):
                path = folder / 'tp-live-in' / claim.name / '000000.npz'
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                if not (fault == 'missing_peer_receipt' and rank == 1):
                    ack = folder / 'tp-live-out' / claim.name / f'ack.000000.rank{rank}.json'
                    ack.write_text(json.dumps({'rank':rank,'world_size':2,'sequence':0,'rows':12,'sha256':digest}))
            active.set()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"text":"PRIVATE_SYNTHETIC_OUTPUT"}]}\n\n')
            if fault == 'natural_completion': self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
            self.connection.settimeout(6)
            try:
                if fault != 'natural_completion' and self.connection.recv(1) == b'': disconnected.set()
            finally:
                active.clear()
                self.close_connection = True
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runner = tmp_path / 'runner.py'
    runner.write_text((ROOT/'scripts/live/spark_live_session.py').read_text().replace('Path("/dev/shm/glm53-handoff")', f'Path({str(rank_roots[0])!r})'))
    messages = claim.path / 'messages.json'
    messages.write_text('[{"role":"user","content":"synthetic input"}]')
    base = f'http://127.0.0.1:{server.server_port}'
    monkeypatch.setenv('DRIFT_GLM_KEY', '')
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    limits = Limits(max_new=256,max_prompt_tokens=256,max_seconds=5,stop_timeout=.1)
    command = [sys.executable,str(runner),'--session',claim.name,'--messages',str(messages),'--base',base,
               '--max-new','256','--reserve','12','--max-prompt-tokens','256','--control-stdin',
               '--wait-publication','--ready-timeout','4','--no-tap']
    client = LinkedSession(command,claim,limits)
    def check():
        client.poll(0)
        if client.request_state().completed: raise RuntimeError('natural completion')
    transport = PublicationTransport(source,digest,claim.name,'peer',check,root=rank_roots[0])
    transport.delivery.timeout = .4
    real_run = subprocess.run
    def local_ranks(argv, **kwargs):
        peer = argv[0] == 'ssh'
        actual = shlex.split(argv[-1]) if peer else list(argv)
        if actual[:2] == ['python3','-c']:
            actual[0] = sys.executable
            actual[2] = actual[2].replace('/dev/shm/glm53-handoff',str(rank_roots[int(peer)]))
        return real_run(actual,**kwargs)
    monkeypatch.setattr(subprocess,'run',local_ranks)
    def sample(*,timeout): return parse_metrics(fetch_metrics(base+'/metrics',timeout=min(timeout,2)))
    try:
        result = drive_linked(CancellationObserver(timeout=4),client,sample,transport,rows=12,digest=digest,
                              max_seconds=6,poll_seconds=.02)
        assert len(requests) == 1 and result['client']['child_terminated']
        assert 'PRIVATE_SYNTHETIC_OUTPUT' not in json.dumps(result)
        if fault is None:
            assert result['status'] == 'PASSED', result
            assert len(result['publication']['receipts']) == 2 and result['client']['cancelled']
            assert result['settled_samples'] == 3 and disconnected.is_set()
        else:
            assert result['status'] == 'INVALID' and not result['client']['cancelled']
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
