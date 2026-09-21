"""Admit complete development-loop evidence without examining model answers."""
import re


class _InvalidEvidence(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise _InvalidEvidence(reason)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _reverse(publications, session):
    for sequence, publication in enumerate(publications):
        _require(type(publication) is dict, "REVERSE_RECORD_INVALID")
        _require(not any(key in publication for key in ("error", "skipped")), "REVERSE_FAILURE")
        _require(publication.get("session") == session, "REVERSE_SESSION_MISMATCH")
        _require(_integer(publication.get("sequence")) and publication["sequence"] == sequence, "REVERSE_SEQUENCE_INVALID")
        _require(_integer(publication.get("rows"), 1) and _integer(publication.get("bytes"), 1), "REVERSE_SIZE_INVALID")
        source = publication.get("source_sha256")
        _require(_digest(source), "REVERSE_SOURCE_DIGEST_MISSING")
        receipts = publication.get("receipts")
        _require(type(receipts) is dict and len(receipts) == 2, "REVERSE_RANK_RECEIPTS_MISSING")
        ranks, applied = [], []
        for receipt in receipts.values():
            _require(type(receipt) is dict, "REVERSE_RECEIPT_INVALID")
            _require(not any(key in receipt for key in ("error", "skipped")), "REVERSE_RECEIPT_INVALID")
            _require(_integer(receipt.get("rank")) and type(receipt.get("world_size")) is int and receipt["world_size"] == 2, "REVERSE_RANK_INVALID")
            _require(_integer(receipt.get("sequence")) and receipt["sequence"] == sequence, "REVERSE_RECEIPT_SEQUENCE_MISMATCH")
            _require(_integer(receipt.get("rows"), 1) and receipt["rows"] == publication["rows"], "REVERSE_RECEIPT_ROWS_MISMATCH")
            _require(receipt.get("source_sha256") == source and _digest(receipt.get("sha256")), "REVERSE_RECEIPT_DIGEST_MISMATCH")
            ranks.append(receipt["rank"])
            applied.append(receipt["sha256"])
        _require(sorted(ranks) == [0, 1], "REVERSE_RANK_INVALID")
        _require(len(set(applied)) == 1, "REVERSE_APPLIED_DIGEST_MISMATCH")


def _forward(taps, session, apply):
    previous_stop = None
    for sequence, tap in enumerate(taps):
        _require(type(tap) is dict, "FORWARD_RECORD_INVALID")
        _require(not any(key in tap for key in ("error", "skipped")), "FORWARD_FAILURE")
        _require(tap.get("session") == session, "FORWARD_SESSION_MISMATCH")
        _require(_integer(tap.get("tap")) and tap["tap"] == sequence, "FORWARD_SEQUENCE_INVALID")
        _require(tap.get("cache_applied") is apply, "FORWARD_APPLICATION_MISMATCH")
        if not apply:
            _require(tap.get("dropped_by_control") is True, "FORWARD_CONTROL_DROP_MISSING")
            continue
        _require(not tap.get("dropped_by_control"), "FORWARD_UNEXPECTED_DROP")
        _require(_integer(tap.get("rows"), 1), "FORWARD_ROWS_INVALID")
        positions = tap.get("glm_positions")
        _require(type(positions) is list and len(positions) == 2 and all(_integer(n) for n in positions), "FORWARD_POSITIONS_INVALID")
        start, stop = positions
        _require(start < stop and (previous_stop is None or start == previous_stop), "FORWARD_POSITIONS_INVALID")
        previous_stop = stop


def _inspect(report, condition):
    _require(condition in ("linked", "no_link", "reverse_only", "forward_only"), "CONDITION_INVALID")
    _require(type(report) is dict, "REPORT_INVALID")
    summary, qwen = report.get("summary"), report.get("qwen")
    _require(type(summary) is dict and type(qwen) is dict, "REPORT_INVALID")
    session = summary.get("session")
    _require(type(session) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session) is not None, "SESSION_MISSING")
    _require(qwen.get("session") == session, "SESSION_MISMATCH")
    no_link, reverse_on = condition == "no_link", condition in ("linked", "reverse_only")
    forward_on = condition in ("linked", "forward_only")
    _require(summary.get("no_link") is no_link and qwen.get("no_link") is no_link, "CONDITION_MISMATCH")
    _require(summary.get("reverse_errors") == [], "REVERSE_FAILURE")
    _require(qwen.get("peer_done") is True, "PEER_COMPLETION_MISSING")
    _require(qwen.get("forward_complete") is (not no_link), "FORWARD_COMPLETION_MISSING")
    epochs, taps = qwen.get("epochs"), qwen.get("forward_taps")
    _require(type(epochs) is list and epochs and type(taps) is list, "EVENTS_MISSING")
    _require(all(type(epoch) is dict and "reverse" in epoch and not any(k in epoch for k in ("error", "skipped")) for epoch in epochs), "EPOCH_INVALID")
    publications = [epoch["reverse"] for epoch in epochs if epoch["reverse"] is not None]
    _require(bool(publications) is reverse_on, "REVERSE_DIRECTION_MISMATCH")
    _require(bool(taps) is (not no_link), "FORWARD_DIRECTION_MISMATCH")
    _reverse(publications, session)
    _forward(taps, session, forward_on)
    _require(_integer(qwen.get("forward_tap_count")) and qwen["forward_tap_count"] == len(taps), "FORWARD_TERMINAL_COUNT_MISMATCH")
    _require(_integer(summary.get("reverse_publications")) and summary["reverse_publications"] == len(publications), "REVERSE_SUMMARY_COUNT_MISMATCH")
    expected_forward = len(taps) if forward_on else 0
    _require(_integer(summary.get("forward_taps")) and summary["forward_taps"] == expected_forward, "FORWARD_SUMMARY_COUNT_MISMATCH")
    glm = summary.get("glm")
    _require(type(glm) is dict and type(glm.get("connector")) is dict, "GLM_COMPLETION_MISSING")
    connector = glm["connector"]
    _require(type(connector.get("failed")) is str and connector["failed"] == "", "GLM_CONNECTOR_FAILED")
    _require(_integer(connector.get("writes_scheduled")) and connector["writes_scheduled"] == len(publications), "GLM_WRITE_COUNT_MISMATCH")
    if not no_link:
        _require(_integer(connector.get('tap_count')) and connector['tap_count'] == len(taps), 'FORWARD_CONNECTOR_COUNT_MISMATCH')
        start, stop = connector.get('source_start'), connector.get('source_stop')
        _require(_integer(start) and _integer(stop) and start <= stop < 2**63, 'FORWARD_TERMINAL_FRONTIER_INVALID')
        _require(len(taps) <= stop - start <= 4096 * len(taps), 'FORWARD_TERMINAL_FRONTIER_INVALID')
        if forward_on:
            _require(start == taps[0]['glm_positions'][0] and stop == taps[-1]['glm_positions'][1],
                     'FORWARD_TERMINAL_FRONTIER_MISMATCH')


def admit_loop_report(report, condition):
    """Return generic validity reasons, never answers or a scientific verdict."""
    try:
        _inspect(report, condition)
    except _InvalidEvidence as error:
        return {"valid": False, "reasons": [str(error)]}
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return {"valid": False, "reasons": ["REPORT_INVALID"]}
    return {"valid": True, "reasons": []}
