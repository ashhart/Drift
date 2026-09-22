"""Verify and retain one tensor-only MCDMA publication before any model can read it."""
import hashlib
from pathlib import Path
import re
import time

from drift.serving.handoffd_client import HandoffdClient, HandoffdError
from drift.serving.worker_activation_files import private_root
from drift.transcript.blob import decode, MAX_BYTES
from drift.transcript.files import publish
from drift.transcript.schema import canonical


def receive(*, peer, remote, sha256, size, output, socket_path, timeout=60, client=None):
    if (type(size) is not int or not 12 < size <= MAX_BYTES or type(sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', sha256) is None
            or type(timeout) not in (int, float) or not 0 < timeout <= 600):
        raise ValueError('TRANSCRIPT_TRANSFER_CONTRACT')
    output = Path(output)
    private_root(output.parent)
    receipt = output.with_suffix(output.suffix + '.receipt.json')
    if any(path.exists() or path.is_symlink() for path in (output, receipt)):
        raise ValueError('TRANSCRIPT_OUTPUT_EXISTS')
    client = client or HandoffdClient(socket_path, timeout_s=timeout)
    started = time.monotonic()
    try:
        pulled = client.pull(peer, remote, offset=0, unlink=False)
    except (HandoffdError, OSError):
        return {'status': 'FAILED', 'error': 'TRANSCRIPT_TRANSPORT_UNAVAILABLE',
                'cache_applied': False, 'native_recall': 'BLOCKED'}
    if pulled.bytes != size:
        raise ValueError('TRANSCRIPT_TRANSFER_SIZE')
    view = client.view(peer, 0, size)
    try:
        blob = bytes(view)
    finally:
        view.release()
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError('TRANSCRIPT_TRANSFER_DIGEST')
    arrays = decode(blob)
    artifact = publish(output, blob)
    report = {'status': 'PASSED', 'scope': 'transport_only', 'transport': 'mcdma',
              'kind': 'transcript_backed_memory', 'activation': artifact,
              'rows': next(iter(arrays.values())).shape[0], 'cache_applied': False,
              'activation_text_bytes': 0, 'activation_token_ids': 0,
              'wall_seconds': time.monotonic() - started, 'rdma_loop_ns': pulled.loop_ns,
              'api_cache_recovered': False, 'native_recall': 'BLOCKED'}
    publish(receipt, (canonical(report) + '\n').encode())
    return report
