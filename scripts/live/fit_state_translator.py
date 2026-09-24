"""Fit a translator from one model's per-token cache features to the other model's recurrent-layer inputs.

Qwen to GLM maps Qwen's full-attention K/V (x) to GLM's KDA inputs (h{layer}); GLM to Qwen maps GLM's MLA latents
(l{layer}) to Qwen's linear-attention inputs (h{layer}). Rows pair a sender token with a receiver token when both end at
the same character of the passage. The sender's features are centred and reduced to their top principal directions;
each receiver layer then gets a ridge map from that subspace to its input. Ridge shrinks its predictions, so each output is also given a gain, the validation
ratio of true to predicted spread, applied at use with a chosen power. Validation R-squared is reported per layer.
With --phases N, sender token i gets its own ridge map for i % N over the shared subspace: DeepSeek V4's four tokens of
one compressed entry share their features, and only a per-position map can tell them apart. drift.translate.ridge_map
loads and applies the result.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True, help="training sender taps, one npz per passage")
parser.add_argument("--target", type=Path, required=True, help="training receiver captures, one npz per passage")
parser.add_argument("--val-source", type=Path, required=True)
parser.add_argument("--val-target", type=Path, required=True)
parser.add_argument("--source-key", default="x", help="'x' for one flat array, or a prefix such as 'l' to join l{layer} arrays in layer order")
parser.add_argument("--target-prefix", default="h", help="receiver layer inputs are {prefix}{layer}; a key that exists as is, such as x, is one output named 0")
parser.add_argument("--rank", type=int, default=2048)
parser.add_argument("--ridge", type=float, default=1.0, help="relative to the mean eigenvalue of the reduced features")
parser.add_argument("--phases", type=int, default=1, help="separate ridge maps by sender token index modulo this")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


def features(z) -> np.ndarray:
    if args.source_key in z.files:
        return z[args.source_key]
    keys = sorted((k for k in z.files if k.startswith(args.source_key) and k[len(args.source_key):].isdigit()), key=lambda k: int(k[len(args.source_key):]))
    return np.concatenate([z[k].reshape(len(z[k]), -1) for k in keys], axis=1)


def pairs(source_dir: Path, target_dir: Path):
    """Aligned (x [n, features], {layer: y [n, width]}, sender token indices [n]) per passage present in both folders."""
    prefix = args.target_prefix
    for path in sorted(target_dir.glob("*.npz")):
        q = source_dir / path.name
        if not q.exists():
            continue
        g, qz = np.load(path), np.load(q)
        ends_g = {int(e): i for i, e in enumerate(g["offsets"][:, 1])}
        shared = [(i, ends_g[int(e)]) for i, e in enumerate(qz["offsets"][:, 1]) if int(e) in ends_g]
        if len(shared) < 4:
            continue
        qi, gi = np.array([a for a, _ in shared]), np.array([b for _, b in shared])
        if prefix in g.files:                                        # one flat receiver array, such as Qwen's full-attention rows
            yield features(qz)[qi].astype(np.float64), {0: g[prefix][gi].reshape(len(gi), -1).astype(np.float64)}, qi
            continue
        yield features(qz)[qi].astype(np.float64), {int(k[len(prefix):]): g[k][gi].astype(np.float64) for k in g.files
                                                     if k.startswith(prefix) and k[len(prefix):].isdigit()}, qi


started = time.time()
count, total, gram, layers = 0, None, None, None
for x, ys, _ in pairs(args.source, args.target):
    total = x.sum(0) if total is None else total + x.sum(0)
    gram = x.T @ x if gram is None else gram + x.T @ x
    count += len(x)
    layers = layers or sorted(ys)
mean = total / count
cov = gram / count - np.outer(mean, mean)
values, vectors = np.linalg.eigh(cov)
basis = vectors[:, ::-1][:, :args.rank]                              # top principal directions
kept = values[::-1][:args.rank]
P = args.phases
zz, zsum, rows_in = np.zeros((P, args.rank, args.rank)), np.zeros((P, args.rank)), np.zeros(P)
zy, ysum, total_y = {l: [None] * P for l in layers}, {l: [None] * P for l in layers}, {l: None for l in layers}
for x, ys, index in pairs(args.source, args.target):
    z = (x - mean) @ basis
    for phase in range(P):
        rows = index % P == phase
        zz[phase] += z[rows].T @ z[rows]
        zsum[phase] += z[rows].sum(0)
        rows_in[phase] += rows.sum()
        for l in layers:
            part, sums = z[rows].T @ ys[l][rows], ys[l][rows].sum(0)
            zy[l][phase] = part if zy[l][phase] is None else zy[l][phase] + part
            ysum[l][phase] = sums if ysum[l][phase] is None else ysum[l][phase] + sums
    for l in layers:
        total_y[l] = ys[l].sum(0) if total_y[l] is None else total_y[l] + ys[l].sum(0)
z_mean = zsum / rows_in[:, None]                                       # each phase is centred on its own rows
y_mean = {l: np.stack([ysum[l][phase] / rows_in[phase] for phase in range(P)]) for l in layers}
out_mean = {l: total_y[l] / count for l in layers}
solves = []
for phase in range(P):
    centred = zz[phase] - rows_in[phase] * np.outer(z_mean[phase], z_mean[phase])
    solves.append(np.linalg.inv(centred + args.ridge * float(np.trace(centred)) / args.rank * np.eye(args.rank)))
weights = {l: np.stack([solves[phase] @ (zy[l][phase] - rows_in[phase] * np.outer(z_mean[phase], y_mean[l][phase])) for phase in range(P)])
           for l in layers}


def predict(z: np.ndarray, index: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Each phase's centred prediction, before its gain and output mean."""
    out = np.empty((len(z), weight.shape[2]))
    for phase in range(P):
        rows = index % P == phase
        out[rows] = (z[rows] - z_mean[phase]) @ weight[phase]
    return out


sse, sst = {l: 0.0 for l in layers}, {l: 0.0 for l in layers}
pred_sq, true_sq, val_rows = {l: 0.0 for l in layers}, {l: 0.0 for l in layers}, 0          # per-output sums, widths from the data
for x, ys, index in pairs(args.val_source, args.val_target):
    z = (x - mean) @ basis
    val_rows += len(x)
    for l in layers:
        pred, centred = predict(z, index, weights[l]), ys[l] - y_mean[l][index % P]
        sse[l] += float(((centred - pred) ** 2).sum())
        sst[l] += float(((ys[l] - out_mean[l]) ** 2).sum())
        pred_sq[l] += (pred ** 2).sum(0)
        true_sq[l] += (centred ** 2).sum(0)
r2 = {l: 1 - sse[l] / sst[l] for l in layers}
gains = {l: np.sqrt(true_sq[l] / np.maximum(pred_sq[l], 1e-12)) for l in layers}
phased = {"phases": np.int32(P), "phase_mean": z_mean.astype(np.float32)} if P > 1 else {}
np.savez(args.out, mean=mean.astype(np.float32), basis=basis.astype(np.float16), layers=np.asarray(layers, np.int32), **phased,
         **{f"W{l}": (weights[l][0] if P == 1 else weights[l]).astype(np.float16) for l in layers},
         **{f"b{l}": (y_mean[l][0] if P == 1 else y_mean[l]).astype(np.float32) for l in layers}, **{f"gain{l}": gains[l].astype(np.float32) for l in layers})
report = {"train_rows": count, "val_rows": val_rows, "rank": args.rank, "ridge": args.ridge, "phases": P, "variance_kept": float(kept.sum() / values.sum()),
          "r2": {str(l): round(r2[l], 4) for l in layers}, "r2_mean": round(float(np.mean(list(r2.values()))), 4), "seconds": round(time.time() - started, 1)}
args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "r2"}))
