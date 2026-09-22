import hashlib
import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drift.serving.bridge_files import GLM_LAYERS
from drift.serving.glm53_handoff import MAGIC, pack_fp8_ds_mla
from drift.transcript.answer import answer
from drift.transcript.blob import decode
from drift.transcript.capture import capture
from drift.transcript.receive import receive
from drift.transcript.schema import Transcript
from test_transcript_schema import conversation


def export_blob(ids, handoff):
    packed = pack_fp8_ds_mla(np.ones((len(ids), 512), np.float32))
    packed = np.pad(packed, ((0, 0), (0, 128)))
    tensors, body = [], bytearray()
    for layer in GLM_LAYERS:
        tensors.append({'layer': f'language_model.model.layers.{layer}.self_attn.attn',
                        'shape': [1, len(ids), 656], 'dtype': 'uint8',
                        'offset': len(body), 'nbytes': packed.nbytes})
        body.extend(packed.tobytes())
    header = json.dumps({'format': 'glm53-handoff-raw-v1', 'cache_config': {'cache_dtype': 'fp8_ds_mla'},
                         'handoff_id': handoff, 'n_tokens': len(ids), 'prompt_token_ids': ids,
                         'tp_rank': 0, 'tensors': tensors}).encode()
    prefix = MAGIC + struct.pack('<Q', len(header)) + header
    return prefix + bytes((-len(prefix)) % 4096) + body


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import drift.transcript.capture as module
    tmp_path.chmod(0o700)
    root = tmp_path / 'exports'
    root.mkdir(mode=0o700)
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_bytes(b'fixture tokenizer')
    tokenized = []
    class Encoder:
        def encode(self, text, **kwargs):
            tokenized.append(text)
            return SimpleNamespace(ids=[4, 8, 15])
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(
        Tokenizer=SimpleNamespace(from_str=lambda text: Encoder())))
    requests = []
    def post(url, body, timeout):
        requests.append(body)
        hid = body['kv_transfer_params']['handoff_id']
        blob = export_blob(body['prompt'], hid)
        destination = root / hid
        (destination / 'rank0.bin').write_bytes(blob)
        (destination / 'rank0.ready').write_text(json.dumps({'mode': 'file', 'bytes': len(blob)}))
        return {'model': 'glm-fixture', 'usage': {'prompt_tokens': 3, 'completion_tokens': 1}}
    monkeypatch.setattr(module, 'request', post)
    options = dict(tokenizer=tokenizer, tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                   model='glm-fixture', export_root=root, output=tmp_path / 'source.memory', timeout=10)
    return options, requests, tokenized


def test_capture_transfer_answer_seam_without_transcript_to_reader(tmp_path, setup):
    options, requests, texts = setup
    report = capture(Transcript.parse(conversation()), **options)
    assert len(requests) == 1 and len(texts) == 1
    assert report['prefill_tokens'] == 3 and report['generated_tokens'] == 1
    blob = options['output'].read_bytes()
    assert b'prompt_token_ids' not in blob and b'18:30' not in blob
    assert set(decode(blob)) == set(GLM_LAYERS)
    class Client:
        def pull(self, peer, remote, **kwargs):
            assert kwargs == {'offset': 0, 'unlink': False}
            return SimpleNamespace(bytes=len(blob), loop_ns=100)
        def view(self, peer, offset, length):
            return memoryview(blob)
    local = tmp_path / 'received.memory'
    transferred = receive(peer='fixture', remote='/memory/source.memory',
                          sha256=report['activation']['sha256'], size=len(blob),
                          output=local, socket_path='/unused', client=Client())
    assert transferred['cache_applied'] is False
    question = tmp_path / 'question.txt'
    question.write_text('What was observed?')
    class Reader:
        def answer(self, latents, question, **kwargs):
            assert set(latents) == set(GLM_LAYERS)
            assert question == 'What was observed?'
            return {'answer': 'fixture result, not recall evidence', 'cache_applied': True}
    result = answer(memory=local, sha256=report['activation']['sha256'], question=question,
                    output=tmp_path / 'answer.json', reader_factory=Reader)
    assert result['transcript_text_bytes_to_reader'] == 0
    assert 'answer' not in result and result['native_recall'] == 'BLOCKED'
    assert options['output'].stat().st_mode & 0o777 == 0o400
    assert json.loads((tmp_path / 'answer.json').read_text())['answer'].startswith('fixture')


@pytest.mark.parametrize('change', ['tokenizer', 'budget', 'output'])
def test_capture_preflight_does_not_start_inference(setup, change):
    options, requests, _ = setup
    if change == 'tokenizer':
        options['tokenizer_sha256'] = '0' * 64
    elif change == 'budget':
        options['max_tokens'] = 2
    else:
        options['output'].write_bytes(b'existing')
    with pytest.raises(ValueError):
        capture(Transcript.parse(conversation()), **options)
    assert requests == []


def test_export_identity_mismatch_cannot_publish(setup, monkeypatch):
    import drift.transcript.capture as module
    options, _, _ = setup
    original = module.request
    def post(url, body, timeout):
        report = original(url, body, timeout)
        directory = options['export_root'] / body['kv_transfer_params']['handoff_id']
        (directory / 'rank0.bin').write_bytes(export_blob(body['prompt'], 'wrong-session'))
        return report
    monkeypatch.setattr(module, 'request', post)
    with pytest.raises(ValueError, match='IDENTITY'):
        capture(Transcript.parse(conversation()), **options)
    assert not options['output'].exists()


def test_bad_transfer_digest_never_publishes(tmp_path, setup):
    options, _, _ = setup
    capture(Transcript.parse(conversation()), **options)
    blob = options['output'].read_bytes()
    client = SimpleNamespace(pull=lambda *a, **k: SimpleNamespace(bytes=len(blob)),
                             view=lambda *a: memoryview(blob))
    output = tmp_path / 'received.memory'
    with pytest.raises(ValueError, match='DIGEST'):
        receive(peer='fixture', remote='/memory/source.memory', sha256='0'*64,
                size=len(blob), output=output, socket_path='/unused', client=client)
    assert not output.exists()


def test_answer_checks_digest_before_loading_model(tmp_path):
    memory, question = tmp_path / 'memory', tmp_path / 'question'
    memory.write_bytes(b'bad')
    question.write_text('Question')
    def forbidden():
        pytest.fail('loaded model before validation')
    with pytest.raises(ValueError, match='MEMORY_PIN'):
        answer(memory=memory, sha256='0'*64, question=question,
               output=tmp_path / 'answer', reader_factory=forbidden)


def test_owned_shared_export_parent_keeps_request_private(setup):
    options, requests, _ = setup
    options['export_root'].chmod(0o755)
    capture(Transcript.parse(conversation()), **options)
    assert len(requests) == 1
    request_dirs = list(options['export_root'].iterdir())
    assert len(request_dirs) == 1
    assert request_dirs[0].stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize('mode', [0o777, 0o775])
def test_writable_export_parent_never_starts_inference(setup, mode):
    options, requests, _ = setup
    options['export_root'].chmod(mode)
    with pytest.raises(ValueError):
        capture(Transcript.parse(conversation()), **options)
    assert not requests
