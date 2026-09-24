"""Scenario generation reproduces earlier runs and balances the answer's position when asked."""
from collections import Counter
import pytest
from scripts.live.appointment_scenarios import VOCABULARY, scenarios

SEED7 = [  # recorded by the selection ablation's run A
    ("Kestrel Coffee is on Tamar Lane. The Copper Kettle is on Quarry Road. Gorse Street Grind is on Whinfell Avenue. Bean Harbour is on Pennywell Close.",
     "I have a doctor's appointment at 2 pm on Friday at the Castlefield surgery on Tamar Lane."),
    ("Bean Harbour is on Alder Street. Kestrel Coffee is on Whinfell Avenue. Thistle and Crumb is on Saltmarsh Row. Gorse Street Grind is on Pennywell Close.",
     "I have a doctor's appointment at 2 pm on Friday at the Castlefield surgery on Alder Street."),
    ("Bean Harbour is on Saltmarsh Row. Kestrel Coffee is on Pennywell Close. The Copper Kettle is on Whinfell Avenue. Thistle and Crumb is on Alder Street.",
     "I have a doctor's appointment at 2 pm on Friday at the Castlefield surgery on Alder Street."),
    ("Gorse Street Grind is on Saltmarsh Row. The Copper Kettle is on Tamar Lane. Bean Harbour is on Quarry Road. Thistle and Crumb is on Alder Street.",
     "I have a doctor's appointment at 2 pm on Friday at the Riverside clinic on Saltmarsh Row."),
    ("Gorse Street Grind is on Saltmarsh Row. Thistle and Crumb is on Quarry Road. Larkspur Roasters is on Whinfell Avenue. Bean Harbour is on Tamar Lane.",
     "I have a doctor's appointment at 4 pm on Friday at the Harbour health centre on Tamar Lane."),
    ("Gorse Street Grind is on Quarry Road. Larkspur Roasters is on Pennywell Close. Thistle and Crumb is on Saltmarsh Row. Bean Harbour is on Alder Street.",
     "I have a doctor's appointment at 2 pm on Friday at the Harbour health centre on Alder Street."),
]


def test_the_default_reproduces_the_recorded_development_scenarios():
    assert [(s["listing"], s["fact"]) for s in scenarios(7, 6)] == SEED7


def test_balance_puts_the_answer_in_each_slot_equally_often_without_changing_the_other_draws():
    plain, balanced = scenarios(20260922, 24, vocabulary="fresh"), scenarios(20260922, 24, balance=True, vocabulary="fresh")
    assert Counter(s["target"] for s in balanced) == {0: 6, 1: 6, 2: 6, 3: 6}
    assert [s["target"] for s in balanced[:4]] != [0, 1, 2, 3] or [s["target"] for s in balanced[4:8]] != [0, 1, 2, 3]
    assert [(s["streets"], s["shops"]) for s in plain] == [(s["streets"], s["shops"]) for s in balanced]
    assert all(s["street"] == s["streets"][s["target"]] and s["street"] in s["fact"] for s in balanced)


def test_the_fresh_vocabulary_shares_no_names_and_no_name_contains_another():
    fresh = [name for pool in VOCABULARY["fresh"] for name in pool]
    dev = [name for pool in VOCABULARY["dev"] for name in pool]
    assert not set(fresh) & set(dev)
    streets = VOCABULARY["fresh"][0]
    others = VOCABULARY["fresh"][1] + VOCABULARY["fresh"][2]
    assert not any(street.lower() in other.lower() for street in streets for other in others)
    assert not any(a != b and a.lower() in b.lower() for a in streets for b in streets)


def test_the_confirmation_vocabulary_is_new_and_scoreable():
    confirm = [name for pool in VOCABULARY["confirm2"] for name in pool]
    earlier = [name for key in ("dev", "fresh") for pool in VOCABULARY[key] for name in pool]
    assert len(set(confirm)) == len(confirm) and not {n.lower() for n in confirm} & {n.lower() for n in earlier}
    for pool in VOCABULARY["confirm2"][:2]:                              # streets and shops are scored by substring
        assert not any(a.lower() in b.lower() for a in pool for b in confirm if a != b)


@pytest.mark.parametrize("vocabulary", ["dev", "fresh", "confirm2"])
def test_every_scenario_lists_four_distinct_streets_and_shops(vocabulary):
    for s in scenarios(3, 12, balance=True, vocabulary=vocabulary):
        assert len(set(s["streets"])) == 4 and len(set(s["shops"])) == 4


def test_an_owner_vocabulary_file_is_used_and_checked(tmp_path):
    import json
    from scripts.live.appointment_scenarios import load_vocabulary
    good = {"streets": ["Aster Row", "Birch Hill", "Cotton Way", "Dunmore Road"], "shops": ["Cup One", "Pour House", "Grindstone", "Latte Lab"],
            "clinics": ["Vale surgery"]}
    path = tmp_path / "names.json"
    path.write_text(json.dumps(good))
    pools = load_vocabulary(path)
    assert {s["street"] for s in scenarios(1, 8, balance=True, vocabulary=pools)} <= set(good["streets"])
    for bad in ({**good, "streets": good["streets"][:3]}, {**good, "shops": ["Birch Hill Cafe", *good["shops"][1:]]},
                {**good, "clinics": ["Cup One"]}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_vocabulary(path)


def test_messages_keep_the_fact_away_from_glm_except_in_the_text_arm():
    from scripts.live.appointment_scenarios import LINK, messages
    scenario = scenarios(7, 1, vocabulary="fresh")[0]
    chats = messages(scenario)
    assert chats["glm"][0]["content"] == LINK and "@@DRIFT@@" in LINK
    assert scenario["listing"] in chats["glm"][1]["content"] and scenario["fact"] not in chats["glm"][1]["content"]
    assert chats["qwen"][1]["content"] == scenario["fact"] and "@@DRIFT@@" not in chats["qwen"][0]["content"]
    assert chats["text"][1]["content"] == f"{scenario['fact']} {chats['glm'][1]['content']}"


def test_appointment_items_carry_each_seeds_chats(tmp_path):
    import json, subprocess, sys
    from pathlib import Path
    from scripts.live.appointment_scenarios import messages
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "items.json"
    subprocess.run([sys.executable, "scripts/live/appointment_items.py", "--seeds", "11", "12", "--count", "3", "--vocabulary", "fresh", "--balance", "--out", str(out)],
                   check=True, capture_output=True, cwd=root)
    items = json.loads(out.read_text())
    assert [item["id"] for item in items] == ["s11x0", "s11x1", "s11x2", "s12x0", "s12x1", "s12x2"]
    expected = scenarios(12, 3, True, "fresh")[1]
    assert items[4]["shop"] == expected["shop"] and items[4]["glm"] == messages(expected)["glm"]
    refused = subprocess.run([sys.executable, "scripts/live/appointment_items.py", "--seeds", "1", "--vocabulary", "confirm2", "--out", str(out)],
                             capture_output=True, cwd=root)
    assert refused.returncode != 0


def test_the_training_vocabulary_is_disjoint_from_every_evaluation_name():
    train = [name for group in VOCABULARY["train"] for name in group]
    others = [name for key, groups in VOCABULARY.items() if key != "train" for group in groups for name in group]
    assert len(set(train)) == len(train)
    assert not [(a, b) for a in train for b in train + others if a != b and a.lower() in b.lower()]
    assert not [(a, b) for a in others for b in train if a.lower() in b.lower()]
