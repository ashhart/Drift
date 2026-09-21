"""Fit the reverse-direction MergeFilter (drift/translate/merge.py): which Qwen tokens end a GLM token?"""
import argparse, json
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, nargs="+", required=True)
parser.add_argument("--qwen", type=Path, nargs="+", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rank", type=int, default=512)
parser.add_argument("--layers", type=int, nargs="+", default=[3, 7, 11], help="token identity is most linear in the early KV layers")
parser.add_argument("--ridge", type=float, default=0.05)
parser.add_argument("--ones", type=int, default=90000)
args = parser.parse_args()
find = lambda rid: next(f / f"{rid}.npz" for f in args.qwen if (f / f"{rid}.npz").exists())
rng = np.random.default_rng(0)
rows = {"fit": ([], []), "val": ([], [])}
records, seen = [], set()
for path in args.corpus:
    for r in json.loads(path.read_text())["records"]:
        if r["split"] == "train" and r["id"] not in seen:
            seen.add(r["id"]); records.append(r)
for n, r in enumerate(records):
    q = np.load(find(r["id"])); T = len(r["ids_qwen"])
    Q = np.concatenate([np.concatenate((q[f"k{l}"].reshape(T, -1), q[f"v{l}"].reshape(T, -1)), 1) for l in args.layers], 1).astype(np.float32)
    ends = np.zeros(T, bool); ends[[qi for _, qi in r["aligned"]]] = True
    pair = np.concatenate((Q, np.concatenate((Q[1:], np.zeros_like(Q[:1])))), axis=1)[:-1]            # the final token has no successor; it is always kept
    take = (~ends[:-1]) | (rng.random(T - 1) < 0.35)
    part = rows["val" if n % 20 == 0 else "fit"]
    part[0].append(pair[take]); part[1].append(ends[:-1][take])
X, y = np.concatenate(rows["fit"][0]), np.concatenate(rows["fit"][1]); Xv, yv = np.concatenate(rows["val"][0]), np.concatenate(rows["val"][1])
d = X.shape[1] // 2
mean = X[:, :d].mean(0)
_, _, vh = torch.linalg.svd(torch.from_numpy(X[rng.choice(len(X), size=min(60000, len(X)), replace=False), :d] - mean), full_matrices=False)
basis = vh[: args.rank].T.numpy()
feats = lambda A: np.concatenate(((A[:, :d] - mean) @ basis, (A[:, d:] - mean) @ basis), axis=1).astype(np.float64)
Z, Zv = torch.from_numpy(feats(X)), torch.from_numpy(feats(Xv))
t = torch.from_numpy(y.astype(np.float64)); weight = torch.where(t < 0.5, 5.0, 1.0).sqrt()[:, None]
zm, tm = Z.mean(0), t.mean()
A = (Z - zm) * weight
w = torch.linalg.solve(A.T @ A + args.ridge * float(torch.diagonal(A.T @ A).mean()) * torch.eye(A.shape[1], dtype=torch.float64), A.T @ ((t - tm)[:, None] * weight))[:, 0]
b = float(tm - zm @ w)
score = (Zv @ w + b).numpy()
report = {"fit_rows": int(len(X)), "non_endpoints_fit": int((~y).sum()), "val_rows": int(len(Xv)), "non_endpoints_val": int((~yv).sum()), "thresholds": {}}
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    keep = score >= th
    report["thresholds"][th] = {"endpoints_kept": float(keep[yv].mean()), "non_endpoints_dropped": float((~keep[~yv]).mean())}
best = max(report["thresholds"], key=lambda th: report["thresholds"][th]["endpoints_kept"] + report["thresholds"][th]["non_endpoints_dropped"])
report.update({"threshold": best, "layers": args.layers, "rank": args.rank})
np.savez(args.out, mean=mean, basis=basis, w=w.float().numpy(), b=np.array(b), meta=np.array(json.dumps(report)))
print(json.dumps(report, indent=1))
