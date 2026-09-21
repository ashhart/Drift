"""Validate collaborator inputs before any publication or cache mutation."""
import hashlib
import re

from drift.exchange.contract import MAX_BYTES, MAX_ROWS, digest, integer, require

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def identity(value):
    require(type(value) is str and NAME.fullmatch(value) is not None, "EXCHANGE_BINDING")


def ranks(values):
    require(type(values) is tuple and 1 <= len(values) <= 16, "EXCHANGE_RANKS")
    for value in values:
        identity(value)
    require(len(set(values)) == len(values), "EXCHANGE_RANKS")


def snapshot(value, copies):
    require(type(value) is dict and set(value) == {"body", "sha256", "rows"}, "EXCHANGE_SNAPSHOT")
    require(type(value["body"]) is bytes, "EXCHANGE_BODY")
    integer(len(value["body"]), 1, MAX_BYTES)
    rows = integer(value["rows"], 1, MAX_ROWS)
    integer(rows * copies, 1, MAX_ROWS)
    digest(value["sha256"])
    require(hashlib.sha256(value["body"]).hexdigest() == value["sha256"], "EXCHANGE_DIGEST")
    return rows * copies


def frontier(tap, previous_stop):
    require(type(tap) is dict and "start" in tap and "stop" in tap, "EXCHANGE_TAP")
    start, stop = integer(tap["start"], 0, MAX_ROWS), integer(tap["stop"], 1, MAX_ROWS)
    require(start < stop and (previous_stop is None or start == previous_stop), "EXCHANGE_TAP")
    return start, stop


def completion(tap, count, first_start, previous_stop):
    require(type(tap) is dict and tap.get('op') == 'complete', 'EXCHANGE_COMPLETE')
    require(integer(tap.get('tap_count'), 0, MAX_ROWS) == count, 'EXCHANGE_COMPLETE')
    start = integer(tap.get('source_start'), 0, MAX_ROWS)
    stop = integer(tap.get('source_stop'), 0, MAX_ROWS)
    require(start <= stop, 'EXCHANGE_COMPLETE')
    require(start == (first_start if count else stop), 'EXCHANGE_COMPLETE')
    require(stop == (previous_stop if count else start), 'EXCHANGE_COMPLETE')
