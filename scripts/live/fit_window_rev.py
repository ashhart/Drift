"""Windowed reverse translator (Qwen -> GLM): every GLM entry is predicted from the current Qwen entry (full stack)
plus the previous `--window` Qwen entries in a reduced basis, so digits that Qwen writes as separate tokens are all
visible when GLM's merged token is produced. Same ridge + variance-restoration recipe as fit_stacked_stream.py.
Reports validation error overall and on MERGED rows (GLM tokens that cover several Qwen tokens) against the base."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, required=True)
parser.add_argument("--glm", type=Path, required=True)
parser.add_argument("--qwen", type=Path, required=True)
parser.add_argument("--base", type=Path, default=Path("local/live/stacked3_rev.npz"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--window", type=int, default=2)
parser.add_argument("--rank", type=int, default=1024)
parser.add_argument("--ridge", type=float, default=0.03)
parser.add_argument("--prev-layers", type=int, default=0, help="use the RAW entries of the first N KV layers for previous tokens instead of a PCA of the whole stack (token identity is most linear early)")
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
records, seen = [], set()
for r in json.loads(args.corpus.read_text())["records"]:
    if r["split"] == "train" and r["id"] not in seen:
        seen.add(r["id"]); records.append(r)
fit, val = [r for i, r in enumerate(records) if i % 20], [r for i, r in enumerate(records) if not i % 20]
base = np.load(args.base); mean_q = base["g_mean"].astype(np.float32)


def stacks(r):
    g, q = np.load(args.glm / f"{r['id']}.npz"), np.load(args.qwen / f"{r['id']}.npz"); T = len(r["ids_qwen"])
    Q = np.concatenate([np.concatenate((q[f"k{l}"].reshape(T, -1), q[f"v{l}"].reshape(T, -1)), 1) for l in QL], 1).astype(np.float32) - mean_q
    G = np.concatenate([g[f"l{l}"] for l in GL], 1).astype(np.float32)
    return Q, G


rng = np.random.default_rng(0)
sample = np.concatenate([(lambda Q: Q[rng.choice(len(Q), size=min(24, len(Q)), replace=False)])(stacks(r)[0]) for r in fit[::2]])
_, _, vh = torch.linalg.svd(torch.from_numpy(sample), full_matrices=False)
basis = vh[: args.rank].T.numpy()                                  # [12288, r]
if args.prev_layers:
    basis = np.eye(12288, dtype=np.float32)[:, : args.prev_layers * 1024]; args.rank = args.prev_layers * 1024
print("basis from", len(sample), "rows", flush=True)


def rows(r):
    Q, G = stacks(r)
    Z = Q @ basis
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    ends = set(qi.tolist())
    feats = [Q[qi]]
    for j in range(1, args.window + 1):
        prev = np.where((qi - j >= 0)[:, None], Z[np.maximum(qi - j, 0)], 0.0)
        feats.append(prev.astype(np.float32))
    merged = np.array([(q - 1 >= 0) and ((q - 1) not in ends) for q in qi])
    return torch.from_numpy(np.concatenate(feats, 1)).double(), torch.from_numpy(G[gi]).double(), merged


d_in, d_out, n, t0 = 12288 + args.window * args.rank, 11 * 512, 0, time.time()
sx, sy = torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_out, dtype=torch.float64)
xx, xy = torch.zeros(d_in, d_in, dtype=torch.float64), torch.zeros(d_in, d_out, dtype=torch.float64)
for start in range(0, len(fit), 96):
    parts = [rows(r) for r in fit[start:start + 96]]
    X, Y = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    n += len(X); sx += X.sum(0); sy += Y.sum(0); xx += X.T @ X; xy += X.T @ Y
    print("accumulated", start + len(parts), "records", n, "rows", round(time.time() - t0), "s", flush=True)
mx, my = sx / n, sy / n
cxx, cxy = xx - n * torch.outer(mx, mx), xy - n * torch.outer(mx, my)
W = torch.linalg.solve(cxx + args.ridge * float(torch.diagonal(cxx).mean()) * torch.eye(d_in, dtype=torch.float64), cxy)
V = [rows(r) for r in val]
Xv, Yv, Mv = torch.cat([p[0] for p in V]) - mx, torch.cat([p[1] for p in V]) - my, np.concatenate([p[2] for p in V])
pred = Xv @ W
W0 = torch.from_numpy(np.concatenate([base[f"W{l}"] for l in GL], axis=1)).double(); b0 = torch.from_numpy(np.concatenate([base[f"b{l}"] for l in GL])).double()
pred0 = (Xv[:, :12288] + mx[:12288]) @ W0 + b0 - my                # the base translator on the same rows (its inputs are already centred by mean_q)
rel = lambda p, rows_: float((p[rows_] - Yv[rows_]).pow(2).mean().sqrt() / Yv[rows_].pow(2).mean().sqrt())
everything, merged = np.ones(len(Mv), bool), Mv
report = {"window": args.window, "rank": args.rank, "ridge": args.ridge, "fit_rows": n, "val_rows": int(len(Mv)), "val_merged_rows": int(merged.sum()),
          "val_rel_rmse": {"all": {"base": rel(pred0, everything), "window": rel(pred, everything)}, "merged": {"base": rel(pred0, merged), "window": rel(pred, merged)},
                           "not_merged": {"base": rel(pred0, ~merged), "window": rel(pred, ~merged)}},
          "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest()}
gain = Yv.std(0) / pred.std(0).clamp_min(1e-8)
np.savez(args.out, basis=basis, input_mean_q=mean_q, feature_mean=mx.float().numpy(), W=W.float().numpy(), b=my.float().numpy(), gain=gain.float().numpy(), meta=np.array(json.dumps(report)))
print(json.dumps(report, indent=1))
