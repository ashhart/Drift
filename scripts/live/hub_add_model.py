"""Add a model to the shared space in one command: fit its maps into and out of the hub, check them, run a drop-in test.

Inputs are exports of one set of text windows, read by the new model and by the hub's anchor, one npz per window under
the same name: the anchor's latents and token offsets from export_passages.py, and the new model's offsets and per-token
rows x from its cache adapter (docs/guides/SHARED_SPACE.md). Tokens of the two that end on the same character are
paired. The new model's rows reduce to their principal directions; ridge maps take them into the hub (the encoder) and
take the hub back to the rows (the decoder), with the decoder's spread restored (drift/translate/spread.py). Held-out
windows give each map's fit. With --test, the new member is paired with another member in the direction given, the
pair is written as an ordinary ridge_map translator, and the test command runs with {translator} replaced by its path;
its exit code and output tail go in the member's report.

  python3 scripts/live/hub_add_model.py --make-anchor glm --rows out/rows_g2q_code.npz --out hub/glm
  python3 scripts/live/hub_add_model.py --name qwen --anchor hub/glm --features out/mix_taps \\
      --anchor-features out/mix_latents --val-features out/code_qwen_val --val-anchor out/code_glm_val --out hub/qwen \\
      --test-from hub/glm --test "python3 gate.py --glm-rows {translator}"
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.translate import hub, ridge_map
from drift.translate.context_pairs import windows
from drift.translate.spread import spread_gain


def paired(anchor_dir: Path, model_dir: Path, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """The anchor's features and the new model's rows at tokens ending on the same character, sampled evenly over windows."""
    names = [p for p in sorted(Path(anchor_dir).glob("*.npz")) if (Path(model_dir) / p.name).exists()]
    per_window, rng = -(-max_rows // max(len(names), 1)), np.random.default_rng(seed)
    anchors, rows, count = [], [], 0
    for _, features, source, target in windows(anchor_dir, model_dir):
        keep = np.sort(rng.choice(len(source), min(per_window, len(source)), replace=False))
        anchors.append(features[source[keep]]); rows.append(np.asarray(target[keep], np.float32)); count += 1
    if not count:
        raise SystemExit(f"no window pairs between {anchor_dir} and {model_dir}")
    return np.concatenate(anchors), np.concatenate(rows), count


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    return round(float(1 - ((pred - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-anchor", help="write the hub's anchor member under this name, from --rows's mean and basis")
    parser.add_argument("--rows", type=Path, help="a ridge_map translator whose reduction defines the hub")
    parser.add_argument("--name")
    parser.add_argument("--anchor", type=Path, help="the anchor member's folder")
    parser.add_argument("--features", type=Path, help="the new model's exports of the windows")
    parser.add_argument("--anchor-features", type=Path, help="the anchor's exports of the same windows")
    parser.add_argument("--val-features", type=Path)
    parser.add_argument("--val-anchor", type=Path)
    parser.add_argument("--rank", type=int, default=2048, help="principal directions kept of the new model's features")
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", help="a drop-in test command; {translator} becomes the paired translator's path")
    parser.add_argument("--test-from", type=Path, help="pair from this member into the new one for the test")
    parser.add_argument("--test-to", type=Path, help="or from the new one into this member")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.make_anchor:
        rows = ridge_map.load(args.rows)
        hub.save(hub.anchor(args.make_anchor, rows["mean"], rows["basis"]), args.out,
                 {"anchor": True, "from": str(args.rows), "sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest()})
        print(json.dumps({"anchor": args.make_anchor, "out": str(args.out)}))
        return
    anchor = hub.load(args.anchor)
    started = time.time()
    a, x, count = paired(args.anchor_features, args.features, args.max_rows, args.seed)
    z = anchor.encode(a)
    in_mean, in_basis = hub.principal(x, min(args.rank, x.shape[1]))
    reduced = (x - in_mean) @ in_basis
    enc_w, enc_b = hub.fit_ridge(reduced, z, args.ridge)
    dec_w, dec_b = hub.fit_ridge(z, x, args.ridge)
    centre, gain = spread_gain(z @ dec_w + dec_b, x)
    member = hub.Member(args.name, in_mean, in_basis, enc_w, enc_b, dec_w, dec_b, gain, centre)
    report = {"windows": count, "rows": int(len(x)), "rank": int(in_basis.shape[1]), "ridge": args.ridge, "fit_seconds": round(time.time() - started, 1)}
    if args.val_features and args.val_anchor:
        va, vx, vcount = paired(args.val_anchor, args.val_features, args.max_rows, args.seed)
        vz = anchor.encode(va)
        report["held_out"] = {"windows": vcount, "rows": int(len(vx)), "encoder_r2": r2(member.encode(vx), vz),
                              "decoder_r2": r2(member.decode(vz), vx), "through_the_hub_r2": r2(anchor.decode(member.encode(vx)), va)}
    hub.save(member, args.out, report)
    if args.test:
        source, target = (hub.load(args.test_from), member) if args.test_from else (member, hub.load(args.test_to))
        translator = args.out / f"pair_{source.name}_to_{target.name}.npz"
        hub.save_translator(hub.pair(source, target), translator)
        command = args.test.replace("{translator}", shlex.quote(str(translator)))
        run = subprocess.run(command, shell=True, capture_output=True, text=True)
        report["test"] = {"pair": f"{source.name} -> {target.name}", "command": command, "exit": run.returncode, "output_tail": run.stdout[-2000:]}
        hub.save(member, args.out, report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
