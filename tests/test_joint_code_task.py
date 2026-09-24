"""The joint code task's scenarios are fresh and its scorer runs the code in isolation."""
from scripts.live.joint_code_task import MARKER, code_block, glm_messages, passes, qwen_messages, recalled, scenarios

GOOD = "```python\nimport os\n\ndef get_api_key():\n    return os.environ['{name}']\n```"


def test_scenarios_draw_distinct_names_deterministically():
    first, again = scenarios(20260923, 16), scenarios(20260923, 16)
    assert first == again and len({s["variable"] for s in first}) == 16
    assert all(s["variable"] in s["fact"] for s in first)


def test_only_the_told_arm_and_qwen_see_the_name():
    s = scenarios(1, 1)[0]
    assert s["variable"] not in str(glm_messages(s, told=False)) and s["variable"] in str(glm_messages(s, told=True))
    assert "@@DRIFT@@" in glm_messages(s, told=False)[0]["content"] and s["variable"] in str(qwen_messages(s))


def test_correct_code_passes_and_wrong_or_missing_code_fails():
    assert passes(GOOD.format(name="TEAL_OTTER_KEY"), "TEAL_OTTER_KEY")
    assert not passes(GOOD.format(name="API_KEY"), "TEAL_OTTER_KEY")
    assert not passes("no code here", "TEAL_OTTER_KEY")
    assert not passes("```python\ndef get_api_key():\n    return 'sentinel-5c1d'\n```", "TEAL_OTTER_KEY")
    assert not passes("```python\nwhile True:\n    pass\n```", "TEAL_OTTER_KEY", timeout=1)


def test_the_last_code_block_is_scored_and_recall_reads_the_marker_line():
    text = f"{MARKER} TEAL_OTTER_KEY\n```python\nx = 1\n```\n" + GOOD.format(name="TEAL_OTTER_KEY")
    assert code_block(text).strip().endswith("return os.environ['TEAL_OTTER_KEY']")
    assert recalled(text, "TEAL_OTTER_KEY") and not recalled("TEAL_OTTER_KEY without the marker", "TEAL_OTTER_KEY")
