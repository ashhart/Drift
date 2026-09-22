"""Prefill a supplied transcript on the owned local GLM server, then strip the export."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import urllib.request
from uuid import uuid4

from drift.serving.worker_activation_files import private_root
from drift.transcript.blob import encode, MAX_ROWS
from drift.transcript.export import extract
from drift.transcript.files import export_parent, publish, read
from drift.transcript.schema import Transcript, canonical


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('TRANSCRIPT_REDIRECT_REFUSED')


def request(url, body, timeout):
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
            or parsed.path != '/v1/completions' or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise ValueError('TRANSCRIPT_LOOPBACK_ONLY')
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('DRIFT_GLM_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['DRIFT_GLM_KEY']
    operation = urllib.request.Request(url, data=canonical(body).encode(), headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(operation, timeout=timeout) as response:
        raw = response.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('TRANSCRIPT_HTTP_LIMIT')
    return json.loads(raw)


def capture(transcript, *, tokenizer, tokenizer_sha256, model, export_root, output,
            url='http://127.0.0.1:8888/v1/completions', timeout=60, max_tokens=MAX_ROWS, arena=None):
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600
            or type(max_tokens) is not int or not 0 < max_tokens <= MAX_ROWS):
        raise ValueError('TRANSCRIPT_BUDGET')
    transcript = Transcript.parse(json.loads(transcript.canonical))
    root = export_parent(export_root)
    output = Path(output)
    private_root(output.parent)
    receipt_path = output.with_suffix(output.suffix + '.receipt.json')
    if output.exists() or output.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError('TRANSCRIPT_OUTPUT_EXISTS')
    tokenizer_bytes = read(tokenizer, 64 * 1048576)
    if hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_sha256:
        raise ValueError('TRANSCRIPT_TOKENIZER_PIN')
    from tokenizers import Tokenizer
    encoder = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    ids = encoder.encode(transcript.render(), add_special_tokens=False).ids
    if not 0 < len(ids) <= max_tokens:
        raise ValueError('TRANSCRIPT_TOKEN_BUDGET')
    started = time.monotonic()
    deadline = started + timeout
    handoff = 'transcript-' + uuid4().hex
    directory = root / handoff
    directory.mkdir(mode=0o700)
    result = request(url, {'model': model, 'prompt': ids, 'max_tokens': 1, 'temperature': 0,
                          'kv_transfer_params': {'glm53_handoff': True, 'handoff_id': handoff,
                                                 'stream': False, 'stream_verify': True}}, timeout)
    usage = result.get('usage', {})
    if (result.get('model') != model or type(usage.get('prompt_tokens')) is not int
            or usage['prompt_tokens'] != len(ids) or type(usage.get('completion_tokens')) is not int
            or not 0 <= usage['completion_tokens'] <= 1):
        raise ValueError('TRANSCRIPT_HTTP_IDENTITY')
    if time.monotonic() >= deadline:
        raise TimeoutError('TRANSCRIPT_CAPTURE_TIMEOUT')
    latents = extract(directory, handoff, ids, deadline, arena=arena)
    blob = encode(latents)
    if time.monotonic() >= deadline:
        raise TimeoutError('TRANSCRIPT_CAPTURE_TIMEOUT')
    artifact = publish(output, blob)
    report = {**transcript.receipt(), 'status': 'PASSED', 'scope': 'capture_only',
              'shadow_model': model, 'tokenizer_sha256': tokenizer_sha256,
              'server_checkpoint_attested': False,
              'prefill_tokens': len(ids), 'generated_tokens': usage['completion_tokens'],
              'activation': artifact, 'activation_text_bytes': 0, 'activation_token_ids': 0,
              'wall_seconds': time.monotonic() - started, 'native_recall': 'BLOCKED',
              'export_id': handoff, 'capture_policy': 'fresh_full_snapshot'}
    publish(receipt_path, (canonical(report) + '\n').encode())
    return report
