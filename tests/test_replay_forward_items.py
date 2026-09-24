"""Replaying a live run's forward follow-ups offline: the saved taps join in order and the item keeps the loop's layout."""
import ast
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/live/replay_forward_items.py"


def constant(path, name):
    tree = ast.parse(path.read_text())
    return next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name)


def test_the_frame_is_the_loops_own():
    assert constant(SCRIPT, "FRAME") == constant(ROOT / "scripts/live/studio_mcdma_loop.py", "FRAME")


def run_folder(tmp_path, stops=(3, 5)):
    run = tmp_path / "run"
    (run / "s0.linked").mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"rows": [{"scenario": 0, "shop": "The Tin Mug"}]}))
    (run / "s0.qwen.json").write_text(json.dumps([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]))
    (run / "s0.linked" / "report.json").write_text(json.dumps({"summary": {"glm": {"text": "RECOMMENDATION: The Tin Mug"}},
                                                               "qwen": {"session": "loop-1", "text": "Answer.", "followup_answer": "unknown"}}))
    taps, start = tmp_path / "out" / "loop-1" / "taps", 0
    taps.mkdir(parents=True)
    for index, stop in enumerate(stops):
        np.savez(taps / f"{index:06d}.npz", start=np.int64(start), stop=np.int64(stop), l3=np.full((stop - start, 512), index, np.float16))
        start = stop
    return run


def replay(tmp_path, run):
    return subprocess.run([sys.executable, str(SCRIPT), "--run", str(run), "--taps-root", str(tmp_path / "out"), "--followup", "Which shop?",
                           "--items", str(tmp_path / "items.json"), "--latents", str(tmp_path / "latents")], capture_output=True, text=True)


def test_a_run_replays_as_one_item_with_its_taps_in_order(tmp_path):
    assert replay(tmp_path, run_folder(tmp_path)).returncode == 0
    item = json.loads((tmp_path / "items.json").read_text())[0]
    assert item["id"] == "r0-replay" and item["live_answer"] == "unknown" and item["answer"] == "The Tin Mug"
    assert item["head_text"] == ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                                 "Answer.<|im_end|>\n<|im_start|>user\nMy partner's message, from our shared memory:\n")
    assert item["tail_text"] == "\n\nWhich shop?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    latents = np.load(tmp_path / "latents" / "r0.npz")["l3"]
    assert latents.shape == (5, 512) and latents[:3].max() == 0 and latents[3:].min() == 1


def test_a_gap_between_saved_taps_is_refused(tmp_path):
    run = run_folder(tmp_path)
    taps = tmp_path / "out" / "loop-1" / "taps"
    np.savez(taps / "000001.npz", start=np.int64(4), stop=np.int64(5), l3=np.zeros((1, 512), np.float16))
    result = replay(tmp_path, run)
    assert result.returncode != 0 and "not contiguous" in result.stderr
