"""The reverse recall scorer applies the preregistered rule and prints no names."""
import json
import runpy
from pathlib import Path

score = runpy.run_path(str(Path(__file__).parents[1] / "scripts/score_reverse_recall.py"))["score"]


def report(pairs, **extra):
    rows = [{"scenario": i, "street": "Secret Street", "position": i % 4,
             "linked": {"valid": True, "glm_named_only_true_street": a}, "no_link": {"valid": True, "glm_named_only_true_street": b}}
            for i, (a, b) in enumerate(pairs)]
    return {"rows": rows, **extra}


def test_a_clear_win_is_supported_and_names_stay_out():
    out = score(report([(True, False)] * 14 + [(True, True)] * 5 + [(False, False)] * 5, held_out_inputs=True))
    assert (out["linked"], out["linked_only"], out["no_link_only"], out["verdict"]) == (19, 14, 0, "SUPPORTED")
    assert out["one_sided_p"] < 0.001 and out["held_out_inputs"] is True and "Secret" not in json.dumps(out)


def test_significance_without_18_of_24_is_partly_and_a_tie_is_not_supported():
    assert score(report([(True, False)] * 8 + [(False, False)] * 16))["verdict"] == "PARTLY"
    assert score(report([(True, False)] * 3 + [(False, True)] * 3 + [(True, True)] * 18))["verdict"] == "NOT SUPPORTED"


def test_any_invalid_arm_makes_the_run_invalid():
    bad = report([(True, False)] * 24)
    bad["rows"][5]["no_link"] = {"valid": False, "reasons": ["X"]}
    assert score(bad)["verdict"] == "INVALID"
