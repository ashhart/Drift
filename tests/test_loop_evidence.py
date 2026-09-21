"""Development evidence admission without model answers or hardware."""
from copy import deepcopy

import pytest

from scripts.live.loop_evidence import admit_loop_report


def example(condition="linked"):
    reverse_on = condition in ("linked", "reverse_only")
    transport_on = condition != "no_link"
    forward_on = condition in ("linked", "forward_only")
    source, applied = "1" * 64, "2" * 64
    reverse = {
        "session": "loop-example", "sequence": 0, "rows": 12, "bytes": 128,
        "source_sha256": source,
        "receipts": {f"host-{rank}": {"rank": rank, "world_size": 2, "sequence": 0,
                     "rows": 12, "source_sha256": source, "sha256": applied} for rank in range(2)},
    } if reverse_on else None
    taps = [{"session": "loop-example", "tap": 0, "cache_applied": forward_on,
             **({"rows": 8, "glm_positions": [100, 108]} if forward_on else {"dropped_by_control": True})}] if transport_on else []
    return {
        "summary": {"session": "loop-example", "no_link": not transport_on,
                    "glm": {"connector": {"failed": "", "writes_scheduled": int(reverse_on),
                                          'tap_count': len(taps), 'source_start': 100, 'source_stop': 108 if taps else 100}},
                    "reverse_errors": [], "reverse_publications": int(reverse_on),
                    "forward_taps": int(forward_on)},
        "qwen": {"session": "loop-example", "no_link": not transport_on,
                 "peer_done": True, "forward_complete": transport_on,
                 "forward_tap_count": len(taps), "epochs": [{"reverse": reverse}], "forward_taps": taps},
    }


@pytest.mark.parametrize("condition", ["linked", "no_link", "reverse_only", "forward_only"])
def test_each_condition_requires_only_its_expected_directions(condition):
    assert admit_loop_report(example(condition), condition) == {"valid": True, "reasons": []}


@pytest.mark.parametrize("mutate", [
    lambda r: r["qwen"].update(session="another-session"),
    lambda r: r["qwen"].update(peer_done=False),
    lambda r: r["qwen"].update(forward_complete=False),
    lambda r: r["qwen"].update(forward_tap_count=2),
    lambda r: r["qwen"].update(forward_taps=[]),
    lambda r: r["qwen"].update(epochs=[]),
    lambda r: r["summary"].update(reverse_errors=[{"error": "private output"}]),
    lambda r: r["summary"]["glm"].update(connector=None),
    lambda r: r["summary"]["glm"]["connector"].update(failed=True),
    lambda r: r["summary"]["glm"]["connector"].update(failed=False),
    lambda r: r["summary"]["glm"]["connector"].update(writes_scheduled=2),
    lambda r: r["summary"].update(reverse_publications=0),
    lambda r: r["summary"].update(forward_taps=0),
])
def test_missing_or_inconsistent_completion_is_invalid(mutate):
    report = example()
    mutate(report)
    result = admit_loop_report(report, "linked")
    assert result["valid"] is False and result["reasons"]
    assert "private output" not in str(result)


@pytest.mark.parametrize("field,value", [
    ("session", "wrong"), ("sequence", 1), ("sequence", True), ("rows", 0),
    ("source_sha256", "not-a-digest"), ("receipts", {}), ("error", "private detail"),
    ("skipped", "private detail"),
])
def test_bad_publication_is_invalid(field, value):
    report = example()
    report["qwen"]["epochs"][0]["reverse"][field] = value
    result = admit_loop_report(report, "linked")
    assert result["valid"] is False
    assert "private detail" not in str(result)


@pytest.mark.parametrize("field,value", [
    ("rank", 0), ("world_size", 3), ("sequence", 1), ("rows", 13),
    ("source_sha256", "3" * 64), ("sha256", "3" * 64),
])
def test_both_rank_receipts_must_bind_the_same_publication(field, value):
    report = example()
    report["qwen"]["epochs"][0]["reverse"]["receipts"]["host-1"][field] = value
    assert admit_loop_report(report, "linked")["valid"] is False


@pytest.mark.parametrize("condition", ["no_link", "forward_only"])
def test_disabled_reverse_direction_cannot_publish(condition):
    report = example(condition)
    report["qwen"]["epochs"] = example()["qwen"]["epochs"]
    assert admit_loop_report(report, condition)["valid"] is False


@pytest.mark.parametrize("condition", ["no_link", "reverse_only"])
def test_disabled_forward_direction_cannot_apply(condition):
    report = example(condition)
    report["qwen"]["forward_taps"] = example()["qwen"]["forward_taps"]
    assert admit_loop_report(report, condition)["valid"] is False


def test_reverse_only_requires_explicit_validated_drop_not_missing_forward_evidence():
    report = example("reverse_only")
    report["qwen"]["forward_taps"][0].pop("dropped_by_control")
    assert admit_loop_report(report, "reverse_only")["valid"] is False


@pytest.mark.parametrize("field,value", [("tap", 1), ("tap", True), ("session", "wrong"), ("cache_applied", False), ("rows", 0)])
def test_forward_taps_require_order_session_and_cache_application(field, value):
    report = example()
    report["qwen"]["forward_taps"][0][field] = value
    assert admit_loop_report(report, "linked")["valid"] is False


def test_duplicate_taps_and_reversed_publications_are_invalid():
    report = example()
    report["qwen"]["forward_taps"] *= 2
    report["qwen"]["forward_tap_count"] = report["summary"]["forward_taps"] = 2
    assert admit_loop_report(report, "linked")["valid"] is False
    report = example()
    report["qwen"]["epochs"] *= 2
    report["summary"]["reverse_publications"] = 2
    assert admit_loop_report(report, "linked")["valid"] is False


@pytest.mark.parametrize("value", [None, [], {}, {"summary": []}, {"summary": {}, "qwen": {"epochs": [None]}}])
def test_malformed_reports_fail_closed_without_revealing_values(value):
    result = admit_loop_report(value, "linked")
    assert result["valid"] is False and result["reasons"]


def test_unknown_condition_and_input_mutation_are_rejected():
    report = example()
    before = deepcopy(report)
    assert admit_loop_report(report, "unknown")["valid"] is False
    assert report == before


@pytest.mark.parametrize('condition', ['linked', 'forward_only', 'reverse_only'])
@pytest.mark.parametrize('field,value', [('tap_count', 0), ('source_start', True), ('source_stop', None),
                                       ('source_stop', 99), ('source_stop', 2**63)])
def test_terminal_frontier_fields_are_required_and_bounded(condition, field, value):
    report = example(condition)
    report['summary']['glm']['connector'][field] = value
    assert not admit_loop_report(report, condition)['valid']


@pytest.mark.parametrize('condition', ['linked', 'forward_only'])
@pytest.mark.parametrize('field,value', [('source_start', 99), ('source_stop', 109)])
def test_terminal_frontier_matches_actual_received_positions(condition, field, value):
    report = example(condition)
    report['summary']['glm']['connector'][field] = value
    assert admit_loop_report(report, condition) == {'valid': False, 'reasons': ['FORWARD_TERMINAL_FRONTIER_MISMATCH']}


def test_legacy_count_only_connector_marker_is_not_complete_evidence():
    report = example()
    report['summary']['glm']['connector'] = {'failed': '', 'writes_scheduled': 1}
    assert not admit_loop_report(report, 'linked')['valid']
