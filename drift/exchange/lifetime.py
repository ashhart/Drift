"""Carry a request's cancellation check through synchronous exchange work."""
from contextlib import contextmanager
from contextvars import ContextVar


_current = ContextVar('exchange_checkpoint', default=None)


def checkpoint():
    check = _current.get()
    if check is not None:
        check()


@contextmanager
def request_scope(check):
    previous = _current.get()
    def combined():
        if previous is not None:
            previous()
        if check is not None:
            check()
    token = _current.set(combined)
    try:
        yield
    finally:
        _current.reset(token)


class CheckedRegion:
    """Check before and after each bounded native transport operation."""

    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def put(self, data, offset=0):
        checkpoint()
        result = self.connection.put(data, offset=offset)
        checkpoint()
        return result

    def get(self, length, offset=0):
        checkpoint()
        result = self.connection.get(length, offset=offset)
        checkpoint()
        return result
