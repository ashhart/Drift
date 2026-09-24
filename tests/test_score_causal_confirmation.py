"""The causal confirmation scorer applies the preregistered rules and never echoes names."""
import pytest
from scripts.score_causal_confirmation import score


def rows(linked, no_link, text, measure="glm_named_only_true_street", valid=True):
    return {"rows": [{"position": i % 4, "linked": {"valid": valid, measure: a}, "no_link": {"valid": True, measure: b},
                      "text": {"valid": True, measure: c}} for i, (a, b, c) in enumerate(zip(linked, no_link, text))]}


def test_all_three_hypotheses_make_it_supported():
    result = score(rows([True] * 22 + [False] * 2, [False] * 18 + [True] * 6, [True] * 23 + [False]), "recall")
    assert (result["linked"], result["text"], result["P1"], result["P2"], result["P3"], result["verdict"]) == (22, 23, True, True, True, "SUPPORTED")


def test_trailing_text_by_two_is_only_partly():
    result = score(rows([True] * 21 + [False] * 3, [False] * 24, [True] * 24), "recall")
    assert result["P1"] and result["P2"] and not result["P3"] and result["verdict"] == "PARTLY"


def test_an_invalid_arm_voids_the_run_and_joint_uses_its_own_bar():
    assert score(rows([True] * 16, [False] * 16, [True] * 16, valid=False), "recall")["verdict"] == "INVALID"
    joint = score(rows([True] * 12 + [False] * 4, [False] * 16, [True] * 13 + [False] * 3, measure="passes"), "joint")
    assert joint["P2"] and joint["P3"] and joint["verdict"] == "SUPPORTED"
    assert "Lane" not in str(joint)
