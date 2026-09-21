"""Fetch loopback metrics in one owned process with a deadline that includes cleanup."""
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

MAX_BYTES = 2 * 1024 * 1024


class MetricFetchError(RuntimeError):
    """The fetch failed without exporting response headers, body or exception payloads."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _validate(url, timeout, max_bytes):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('metrics timeout must be finite and positive')
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError('metrics byte limit is invalid')
    try:
        target = urllib.parse.urlsplit(url)
        valid = (target.scheme == 'http' and target.hostname == '127.0.0.1'
                 and not target.username and not target.password and target.path == '/metrics'
                 and not target.query and not target.fragment and target.port != 0)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError('only explicit loopback metrics endpoints are allowed')


def _cleanup(child, deadline):
    try:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=max(0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        raise MetricFetchError('metrics fetch cleanup unconfirmed') from None
    finally:
        if child.stdout is not None:
            child.stdout.close()


def fetch_metrics(url, *, timeout, max_bytes=MAX_BYTES):
    """Keep socket trickling inside a parent deadline and reap the owned fetch process."""
    _validate(url, timeout, max_bytes)
    deadline = time.monotonic() + timeout
    cleanup_reserve = min(0.2, timeout / 5)
    command = [sys.executable, '-I', '-S', str(Path(__file__).resolve()),
               url, str(timeout), str(max_bytes)]
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, env={})
    except OSError:
        raise MetricFetchError('metrics fetch failed') from None
    try:
        try:
            payload, _ = child.communicate(timeout=max(0, deadline - cleanup_reserve - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise TimeoutError('metrics fetch deadline exceeded') from None
        except OSError:
            raise MetricFetchError('metrics fetch failed') from None
        if child.returncode != 0 or len(payload) > max_bytes:
            raise MetricFetchError('metrics fetch failed')
        try:
            result = payload.decode('utf-8')
        except UnicodeDecodeError:
            raise MetricFetchError('metrics fetch failed') from None
    finally:
        _cleanup(child, deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError('metrics fetch deadline exceeded')
    return result


def _worker(url, timeout, max_bytes):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(url, timeout=timeout) as response:
            payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            return 1
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        return 1


if __name__ == '__main__':
    raise SystemExit(_worker(sys.argv[1], float(sys.argv[2]), int(sys.argv[3])))
