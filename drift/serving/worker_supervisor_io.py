"""Relay private worker bytes with bounded queues while observing control and deadline."""
import json
import os
import selectors
import time


class RelayStop(RuntimeError):
    pass


def relay(child, control_fd, input_fd, output_fd, deadline, stopped, maximum, total_limit, counts):
    incoming, outgoing, control = bytearray(), bytearray(), bytearray()
    stdout_open = True
    with selectors.DefaultSelector() as watch:
        watch.register(control_fd, selectors.EVENT_READ, 'control')
        watch.register(child.stdout, selectors.EVENT_READ, 'stdout')
        if input_fd is not None:
            watch.register(input_fd, selectors.EVENT_READ, 'input')
            os.set_blocking(child.stdin.fileno(), False)
        if output_fd is not None:
            os.set_blocking(output_fd, False)
        while True:
            if stopped(): raise RelayStop('signal')
            if time.monotonic() >= deadline: raise RelayStop('deadline')
            if not stdout_open and not outgoing and child.poll() is not None:
                return 'child_exit'
            for key, _ in watch.select(timeout=min(0.05, max(0, deadline - time.monotonic()))):
                kind = key.data
                if kind in ('stdin_write', 'output_write'):
                    buffer = incoming if kind == 'stdin_write' else outgoing
                    try: written = os.write(key.fd, buffer)
                    except BlockingIOError: continue
                    except OSError: raise RelayStop('peer_disconnected') from None
                    del buffer[:written]
                    if not buffer: watch.unregister(key.fileobj)
                    continue
                try: block = os.read(key.fd, 4096)
                except OSError: raise RelayStop('peer_disconnected') from None
                if kind == 'control':
                    if not block: raise RelayStop('controller_eof')
                    control.extend(block)
                    if len(control) > 256: raise RelayStop('control_protocol')
                    if b'\n' in control:
                        try: command = json.loads(control)
                        except (ValueError, UnicodeError): raise RelayStop('control_protocol') from None
                        if command != {'op': 'abort'}: raise RelayStop('control_protocol')
                        raise RelayStop('control_abort')
                elif kind == 'input':
                    if not block: raise RelayStop('input_eof')
                    counts['input_bytes'] += len(block)
                    if not incoming: watch.register(child.stdin, selectors.EVENT_WRITE, 'stdin_write')
                    incoming.extend(block)
                elif kind == 'stdout':
                    if not block:
                        watch.unregister(key.fileobj); stdout_open = False
                        continue
                    counts['output_bytes'] += len(block)
                    if output_fd is not None:
                        if not outgoing: watch.register(output_fd, selectors.EVENT_WRITE, 'output_write')
                        outgoing.extend(block)
                if len(incoming) > maximum or len(outgoing) > maximum or sum(counts.values()) > total_limit:
                    raise RelayStop('io_limit')
