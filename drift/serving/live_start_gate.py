"""Prepare fresh linked-session folders and wait for a bounded typed release."""
import json
import os
import re
import selectors
import time


def prepare_fresh(root, session):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', session) or session in {'.', '..'}:
        raise ValueError('invalid linked session')
    folders = [root / kind / session for kind in ('tp-live-in', 'tp-live-out')]
    if any(folder.exists() or folder.is_symlink() for folder in folders):
        raise ValueError('linked session already exists')
    for folder in folders:
        folder.mkdir(parents=True, mode=0o700)


def wait_release(fd, timeout):
    deadline, frame = time.monotonic() + timeout, bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError('publication release deadline exceeded')
            chunk = os.read(fd, 257)
            frame.extend(chunk)
            if not chunk or len(frame) > 256:
                raise ValueError('invalid publication release')
            if b'\n' in frame:
                try:
                    command = json.loads(frame)
                except (ValueError, UnicodeError):
                    raise ValueError('invalid publication release') from None
                if command != {'op': 'release'} or time.monotonic() >= deadline:
                    raise ValueError('invalid publication release')
                return
