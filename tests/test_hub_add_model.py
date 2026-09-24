"""The one-command recipe fits a new member from paired window exports, reports held-out fit and runs the drop-in test."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from drift.translate import hub, ridge_map

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/live/hub_add_model.py"


def _exports(folder: Path, rng, anchor_mix, model_mix, windows: int, tokens: int = 40) -> None:
    """Anchor windows as export_passages.py writes them and new-model windows as a cache adapter writes them, one hidden source."""
    for side in ("anchor", "model"):
        (folder / side).mkdir(parents=True)
    offsets = np.stack([np.arange(tokens), np.arange(1, tokens + 1)], 1).astype(np.int32)
    for w in range(windows):
        hidden = rng.standard_normal((tokens, 6))
        latents = hidden @ anchor_mix
        np.savez(folder / f"anchor/w{w}.npz", offsets=offsets, aligned=np.array(True), l3=latents[:, :10].astype(np.float16), l7=latents[:, 10:].astype(np.float16))
        np.savez(folder / f"model/w{w}.npz", offsets=offsets, x=(hidden @ model_mix + 0.01 * rng.standard_normal((tokens, 12))).astype(np.float16))


def test_the_recipe_fits_a_member_that_generalises_and_runs_the_test_on_the_paired_translator(tmp_path):
    rng = np.random.default_rng(0)
    anchor_mix, model_mix = rng.standard_normal((6, 20)), rng.standard_normal((6, 12))
    _exports(tmp_path / "train", rng, anchor_mix, model_mix, windows=30)
    _exports(tmp_path / "val", rng, anchor_mix, model_mix, windows=5)
    latents = np.concatenate([np.concatenate([np.load(p)["l3"], np.load(p)["l7"]], 1) for p in sorted((tmp_path / "train/anchor").glob("*.npz"))]).astype(np.float32)
    mean, basis = hub.principal(latents, 6)
    np.savez(tmp_path / "rows.npz", mean=mean, basis=basis, layers=np.array([0], np.int32), W0=np.zeros((6, 1), np.float32), b0=np.zeros(1, np.float32), gain0=np.ones(1, np.float32))
    subprocess.run([sys.executable, str(SCRIPT), "--make-anchor", "glm", "--rows", str(tmp_path / "rows.npz"), "--out", str(tmp_path / "hub/glm")], check=True, capture_output=True)
    test = f"{sys.executable} -c \"import sys; print('ran', sys.argv[1])\" {{translator}}"
    run = subprocess.run([sys.executable, str(SCRIPT), "--name", "qwen", "--anchor", str(tmp_path / "hub/glm"), "--features", str(tmp_path / "train/model"),
                          "--anchor-features", str(tmp_path / "train/anchor"), "--val-features", str(tmp_path / "val/model"), "--val-anchor", str(tmp_path / "val/anchor"),
                          "--rank", "6", "--ridge", "1e-4", "--out", str(tmp_path / "hub/qwen"), "--test-from", str(tmp_path / "hub/glm"), "--test", test],
                         check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "hub/qwen/member.json").read_text())
    assert report["windows"] == 30 and report["held_out"]["windows"] == 5
    assert min(report["held_out"][k] for k in ("encoder_r2", "decoder_r2", "through_the_hub_r2")) > 0.99, run.stdout
    pair = tmp_path / "hub/qwen/pair_glm_to_qwen.npz"
    assert report["test"]["exit"] == 0 and str(pair) in report["test"]["output_tail"]
    member, glm = hub.load(tmp_path / "hub/qwen"), hub.load(tmp_path / "hub/glm")
    x = latents[:7]
    np.testing.assert_allclose(ridge_map.apply(ridge_map.load(pair), x)[0], member.decode(glm.encode(x)), rtol=1e-4, atol=1e-4)
