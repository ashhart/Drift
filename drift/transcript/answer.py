"""Answer from a verified latent artifact without reading the API transcript or its receipt."""
import hashlib
import math
from pathlib import Path
import time

from drift.transcript.blob import decode, MAX_BYTES
from drift.transcript.files import publish, read
from drift.transcript.schema import canonical
from drift.serving.worker_activation_files import private_root


def answer(*, memory, sha256, question, output, reader_factory, max_new=64, max_rows=8192, timeout=60):
    if (type(max_new) is not int or not 0 < max_new <= 512 or type(max_rows) is not int
            or not 0 < max_rows <= 16384 or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0 < timeout <= 600):
        raise ValueError('TRANSCRIPT_READER_BUDGET')
    if Path(output).exists() or Path(output).is_symlink():
        raise ValueError('TRANSCRIPT_OUTPUT_EXISTS')
    private_root(Path(output).parent)
    question = read(question, 16384).decode('utf-8')
    if not question.strip():
        raise ValueError('TRANSCRIPT_QUESTION_EMPTY')
    started = time.monotonic()
    blob = read(memory, MAX_BYTES)
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError('TRANSCRIPT_MEMORY_PIN')
    latents = decode(blob)
    reader = reader_factory()
    result = reader.answer(latents, question, max_new=max_new, max_rows=max_rows, deadline=started + timeout)
    if time.monotonic() >= started + timeout:
        raise TimeoutError('TRANSCRIPT_READER_TIMEOUT')
    report = {**result, 'status': 'PASSED', 'scope': 'execution_only',
              'kind': 'transcript_backed_memory', 'api_cache_recovered': False,
              'memory_sha256': sha256, 'question_text_bytes': len(question.encode()),
              'transcript_text_bytes_to_reader': 0, 'native_recall': 'BLOCKED',
              'wall_seconds': time.monotonic() - started}
    publish(output, (canonical(report) + '\n').encode())
    return {key: value for key, value in report.items() if key != 'answer'}
