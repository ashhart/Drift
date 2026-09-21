"""Supervise one session child and close its HTTP connection on controller cancellation."""
import json
import os
import selectors
import signal
import subprocess


class SessionCancelled(RuntimeError):
    """The controller cancelled or disconnected from its session."""


class SessionControlError(RuntimeError):
    """Control or child output violated the bounded session protocol."""


def _read_control(fd):
    data = os.read(fd, 257)
    if not data:
        raise SessionCancelled("controller disconnected")
    if len(data) > 256:
        raise SessionControlError("control frame exceeds 256 bytes")
    return data


def _stop_child(child, timeout):
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=timeout)
    child.stdout.close()
    if child.stdin is not None:
        child.stdin.close()


def _cancel_signal(signum, frame):
    raise SessionCancelled("supervisor received a termination signal")


def supervise(command, control_fd, emit, *, stop_timeout=2, max_output_line=65536, allow_release=False):
    """Forward bounded JSON events and commit done only after successful child exit."""
    child = subprocess.Popen(command, stdin=subprocess.PIPE if allow_release else subprocess.DEVNULL, stdout=subprocess.PIPE)
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, _cancel_signal)
        with selectors.DefaultSelector() as selector:
            selector.register(control_fd, selectors.EVENT_READ, "control")
            selector.register(child.stdout, selectors.EVENT_READ, "output")
            _relay(selector, child, emit, max_output_line, stop_timeout, allow_release)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            _stop_child(child, stop_timeout)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _relay(selector, child, emit, max_output_line, stop_timeout, allow_release):
    control = bytearray()
    output = bytearray()
    done = None
    output_open = True
    release_ready = released = False
    while output_open:
        ready = selector.select(timeout=0.25)
        for key, _ in ready:
            if key.data == "control":
                control.extend(_read_control(key.fd))
                if len(control) > 256:
                    raise SessionControlError("control frame exceeds 256 bytes")
                if b"\n" in control:
                    try:
                        command = json.loads(control)
                    except (ValueError, UnicodeError):
                        raise SessionControlError("invalid control frame") from None
                    if command == {"op": "abort"}:
                        raise SessionCancelled("controller aborted session")
                    if command != {"op": "release"} or not allow_release or not release_ready or released:
                        raise SessionControlError("unsupported control operation")
                    try:
                        child.stdin.write(b'{"op":"release"}\n')
                        child.stdin.flush()
                    except (OSError, ValueError):
                        raise SessionControlError("publication release failed") from None
                    released = True
                    control.clear()
        for key, _ in ready:
            if key.data != "output":
                continue
            chunk = os.read(key.fd, min(max_output_line + 1, 4096))
            if not chunk:
                selector.unregister(key.fileobj)
                output_open = False
                break
            output.extend(chunk)
            while b"\n" in output:
                raw, _, rest = output.partition(b"\n")
                output = bytearray(rest)
                if len(raw) > max_output_line or done is not None:
                    raise SessionControlError("invalid child output frame")
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise SessionControlError("invalid child output JSON") from None
                if not isinstance(event, dict):
                    raise SessionControlError("child output must be an object")
                if event.get("publication_ready") is True:
                    if not allow_release or release_ready or event != {"publication_ready": True}:
                        raise SessionControlError("invalid publication readiness event")
                    release_ready = True
                elif allow_release and not released:
                    raise SessionControlError("child generated before publication release")
                if event.get("done") is True:
                    done = event
                else:
                    emit(event)
            if len(output) > max_output_line:
                raise SessionControlError("child output frame exceeds capacity")
    if output or child.wait(timeout=stop_timeout) != 0 or done is None:
        raise SessionControlError("session child did not complete successfully")
    if selector.select(timeout=0):
        raise SessionCancelled("controller closed or cancelled before completion")
    emit(done)
