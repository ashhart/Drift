"""Deliver private request input without a blocking pipe write escaping its deadline."""
import os
import selectors
import time


def write_input(child, payload, deadline):
    fd, offset = child.stdin.fileno(), 0
    os.set_blocking(fd, False)
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_WRITE)
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('owned HTTP input deadline')
            if not selector.select(min(.1, remaining)):
                continue
            try:
                offset += os.write(fd, payload[offset:offset + 16384])
            except BlockingIOError:
                continue
    child.stdin.close()
