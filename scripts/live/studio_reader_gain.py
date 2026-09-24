"""Fit a contextual reader's spread gain on the Studio: its rows get Qwen's own spread back (drift/translate/spread.py).

The reader is trained on squared error, which shrinks its rows toward their mean. This reads the training windows with
the trained reader, compares its rows at aligned positions with Qwen's own, and saves the per-dimension gain into a copy
of the reader's folder; the original folder is left as it was. The held-out windows are only measured, never fitted.

  PYTHONPATH=. $OMLX_PY scripts/live/studio_reader_gain.py --reader out/context_reader1 --glm out/corpus_latents \\
      --qwen out/corpus_taps --val-glm out/code_glm_val --val-qwen out/code_qwen_val --out out/context_reader1g
"""
import argparse, json, random, shutil
from pathlib import Path
import numpy as np
from drift.serving.studio_guard import acquire
from drift.translate.context_pairs import windows
from drift.translate.context_reader import ContextRows
from drift.translate.spread import restore, spread_gain

parser = argparse.ArgumentParser()
parser.add_argument("--reader", type=Path, required=True)
parser.add_argument("--glm", type=Path, required=True, help="training windows: GLM exports with offsets")
parser.add_argument("--qwen", type=Path, required=True, help="training windows: Qwen taps with offsets and x")
parser.add_argument("--val-glm", type=Path, required=True)
parser.add_argument("--val-qwen", type=Path, required=True)
parser.add_argument("--windows", type=int, default=160, help="training windows sampled to fit the gain")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
acquire("studio_reader_gain.py", need_gb=24)
if (args.reader / "gain.npz").exists():
    raise SystemExit("the reader already has a gain; fit on the folder without one")
reader = ContextRows(args.reader)


def paired(glm_dir, qwen_dir, keep=None):
    """The reader's rows before any gain and Qwen's own rows, at aligned positions of each window."""
    found = list(windows(glm_dir, qwen_dir))
    if keep is not None:
        found = random.Random(args.seed).sample(found, min(keep, len(found)))
    predictions = [reader.read(features, spread=False)[source] for _, features, source, _ in found]
    targets = [rows.astype(np.float32) for _, _, _, rows in found]
    return np.concatenate(predictions), np.concatenate(targets), len(found)


def fit(predictions, targets):
    """R-squared and the median spread ratio to Qwen's rows."""
    return {"r2": round(float(1 - ((predictions - targets) ** 2).sum() / ((targets - targets.mean(0)) ** 2).sum()), 4),
            "spread_ratio_median": round(float(np.median(predictions.std(0) / np.maximum(targets.std(0), 1e-6))), 4)}


train_p, train_t, train_n = paired(args.glm, args.qwen, args.windows)
centre, gain = spread_gain(train_p, train_t)
val_p, val_t, val_n = paired(args.val_glm, args.val_qwen)
report = {"reader": str(args.reader), "fit_windows": train_n, "fit_rows": len(train_p), "val_windows": val_n, "val_rows": len(val_p),
          "gain_median": round(float(np.median(gain)), 4), "val_before": fit(val_p, val_t), "val_after": fit(restore(val_p, centre, gain), val_t)}
if args.out.exists():
    raise SystemExit(f"{args.out} exists")
shutil.copytree(args.reader, args.out)
np.savez(args.out / "gain.npz", centre=centre, gain=gain)
(args.out / "gain.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report))
