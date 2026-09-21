"""Translate selector keys too (roadmap D3): GLM's stacked latents -> Qwen3.8's RAW sparse-selector keys, per KV layer.
Same recipe as the K/V translator: ridge from streaming second moments on aligned rows, per-dimension variance gain measured
on validation rows (the selector scores blocks with relu(q.k), so shrunken keys would be under-selected)."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, default=Path("local/live/corpus23.json"))
parser.add_argument("--glm", type=Path, default=Path("local/live/taps23_glm"))
parser.add_argument("--index", type=Path, default=Path("local/live/taps_qwen_index"))
parser.add_argument("--base", type=Path, default=Path("local/live/stacked3.npz"))
parser.add_argument("--out", type=Path, default=Path("local/live/index3.npz"))
parser.add_argument("--ridge", type=float, default=0.03)
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
g_mean = torch.from_numpy(np.load(args.base)["g_mean"]).double()
records, seen = [], set()
for r in json.loads(args.corpus.read_text())["records"]:
    if r["split"] == "train" and r["id"] not in seen and (args.index / f"{r['id']}.npz").exists():
        seen.add(r["id"]); records.append(r)
fit, val = [r for i, r in enumerate(records) if i % 20], [r for i, r in enumerate(records) if not i % 20]


def rows(r):
    g, q = np.load(args.glm / f"{r['id']}.npz"), np.load(args.index / f"{r['id']}.npz")
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    G = torch.from_numpy(np.concatenate([g[f"l{l}"][gi] for l in GL], 1).astype(np.float64)) - g_mean
    I = torch.from_numpy(np.concatenate([q[f"i{l}"][qi] for l in QL], 1).astype(np.float64))
    return G, I


first = rows(fit[0]); d_in, d_out = first[0].shape[1], first[1].shape[1]; dim = d_out // len(QL)
n, t0 = 0, time.time()
sy, xx, xy = torch.zeros(d_out, dtype=torch.float64), torch.zeros(d_in, d_in, dtype=torch.float64), torch.zeros(d_in, d_out, dtype=torch.float64)
for start in range(0, len(fit), 96):
    parts = [rows(r) for r in fit[start:start + 96]]
    X, Y = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    n += len(X); sy += Y.sum(0); xx += X.T @ X; xy += X.T @ Y
    print("accumulated", start + len(parts), "records", n, "rows", round(time.time() - t0), "s", flush=True)
my = sy / n                                                       # inputs are already centred with the K/V translator's mean
W = torch.linalg.solve(xx + args.ridge * float(torch.diagonal(xx).mean()) * torch.eye(d_in, dtype=torch.float64), xy - torch.outer(xx.new_zeros(d_in), my))
V = [rows(r) for r in val]
Xv, Yv = torch.cat([p[0] for p in V]), torch.cat([p[1] for p in V]) - my
pred = Xv @ W
rel = [float((pred[:, i * dim:(i + 1) * dim] - Yv[:, i * dim:(i + 1) * dim]).pow(2).mean().sqrt() / Yv[:, i * dim:(i + 1) * dim].pow(2).mean().sqrt()) for i in range(len(QL))]
gain = Yv.std(0) / pred.std(0).clamp_min(1e-8)
meta = {"index_dim": dim, "fit_rows": n, "val_rows": int(len(Xv)), "val_rel_rmse": rel, "ridge": args.ridge, "median_gain": float(gain.median()),
        "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest()}
np.savez(args.out, W=W.float().numpy(), b=my.float().numpy(), gain=gain.float().numpy(), meta=np.array(json.dumps(meta)))
print(json.dumps(meta))
