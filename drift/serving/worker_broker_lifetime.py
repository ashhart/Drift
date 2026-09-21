"""Close owner-held channels and require cleanup evidence within the absolute deadline."""
import json
import os
import select
import shutil
import subprocess
import time


class BrokerWorker:
    def __init__(self, process, control, evidence, directory, config_path, config_sha256, activation, deadline, cleanup):
        self.process, self.control, self.evidence = process, control, evidence
        self.directory, self.config_path, self.config_sha256 = directory, config_path, config_sha256
        self.activation, self.deadline, self.cleanup = activation, deadline, cleanup
        self.receipt = None
        self.closed = False

    def _abort(self):
        if self.control is not None:
            try: os.write(self.control, b'{"op":"abort"}\n')
            except OSError: pass
            os.close(self.control); self.control = None

    def _remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0: raise RuntimeError('BROKER_CLEANUP_UNCONFIRMED')
        return remaining

    def _collect(self):
        raw = bytearray()
        os.set_blocking(self.evidence, False)
        while True:
            if not select.select([self.evidence], [], [], self._remaining())[0]:
                raise RuntimeError('BROKER_CLEANUP_UNCONFIRMED')
            block = os.read(self.evidence, 4097 - len(raw))
            if not block: break
            raw.extend(block)
            if len(raw) > 4096: raise RuntimeError('BROKER_RECEIPT_LIMIT')
        self._remaining()
        receipt = json.loads(raw)
        if receipt.get('event') != 'worker_terminated' or receipt.get('child_reaped') is not True or receipt.get('process_group_alive') is not False:
            raise RuntimeError('BROKER_CLEANUP_UNCONFIRMED')
        return receipt

    def finish(self, *, abort=False):
        if self.closed:
            if self.receipt is None: raise RuntimeError('BROKER_CLEANUP_UNCONFIRMED')
            return self.receipt
        try:
            if self.process.stdin is not None and self.process.stdin.closed:
                self.process.stdin = None
            if abort: self._abort()
            first = max(0, self._remaining() - self.cleanup)
            try:
                self.process.communicate(timeout=first)
            except subprocess.TimeoutExpired:
                self._abort()
                try: self.process.communicate(timeout=self._remaining())
                except subprocess.TimeoutExpired:
                    raise RuntimeError('BROKER_CLEANUP_UNCONFIRMED') from None
            self.receipt = self._collect()
            return self.receipt
        finally:
            self.closed = True
            if self.activation is not None: self.activation.close()
            if self.control is not None: os.close(self.control); self.control = None
            os.close(self.evidence)
            if self.receipt is not None: shutil.rmtree(self.directory)

    def __enter__(self): return self

    def __exit__(self, kind, value, traceback): self.finish(abort=kind is not None)
