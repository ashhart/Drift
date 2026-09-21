"""Fit the context-augmented forward translator (drift/translate/context.py) from streaming second moments, two passes:
pass 1 -> input mean and a PCA subspace of the writer's stacked entries; pass 2 -> ridge on [entry, causal summaries]."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch
from drift.translate.context import causal_summaries

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, required=True)
parser.add_argument("--glm", type=Path, required=True)
parser.add_argument("--qwen", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rank", type=int, default=1024)
parser.add_argument("--decays", type=float, nargs="*", default=[0.6, 0.9, 0.97])
parser.add_argument("--ridge", type=float, nargs="+", default=[0.003, 0.01, 0.03])
parser.add_argument("--val-every", type=int, default=20)
parser.add_argument("--device", default="cpu", help="'mps': batch products in float32 on the GPU, accumulated in float64 on the CPU")
parser.add_argument("--val-source", default="multifact", help="report validation error separately for records whose source starts with this")
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
records, seen = [], set()
for r in json.loads(args.corpus.read_text())["records"]:
    if r["split"] == "train" and r["id"] not in seen:
        seen.add(r["id"]); records.append(r)
fit, val = [r for i, r in enumerate(records) if i % args.val_every], [r for i, r in enumerate(records) if not i % args.val_every]
load_g = lambda r: np.concatenate([np.load(args.glm / f"{r['id']}.npz")[f"l{l}"] for l in GL], 1).astype(np.float32)
t0 = time.time()


def load_q(r, qi):
    q = np.load(args.qwen / f"{r['id']}.npz")
    return np.concatenate([np.concatenate((q[f"k{l}"][qi].reshape(len(qi), -1), q[f"v{l}"][qi].reshape(len(qi), -1)), 1) for l in QL], 1).astype(np.float32)


D = 11 * 512
n, s, gg = 0, torch.zeros(D, dtype=torch.float64), torch.zeros(D, D, dtype=torch.float64)
for start in range(0, len(fit), 96):                                          # pass 1: every writer row (summaries see all of them)
    G = torch.from_numpy(np.concatenate([load_g(r) for r in fit[start:start + 96]])).double()
    n += len(G); s += G.sum(0); gg += G.T @ G
mean = s / n
cov = gg / n - torch.outer(mean, mean)
evals, evecs = torch.linalg.eigh(cov)
P = evecs[:, -args.rank:].flip(1).float().numpy() if args.decays else np.zeros((D, 0), np.float32)
mean32 = mean.float().numpy()
summary_scale = float(np.sqrt(float(torch.diagonal(cov).mean()) / max(1e-12, float(evals[-args.rank:].mean())))) if args.decays else 1.0
print("pass 1:", n, "writer rows, PCA keeps", round(float(evals[-args.rank:].sum() / evals.sum()), 3), "of the variance, summary scale", round(summary_scale, 3), round(time.time() - t0), "s", flush=True)


def rows(r):
    x = load_g(r) - mean32
    f = np.concatenate((x, causal_summaries(x @ P, args.decays)[0] * summary_scale), 1) if args.decays else x
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    return torch.from_numpy(f[gi]).double(), torch.from_numpy(load_q(r, qi)).double()


F, O = D + P.shape[1] * len(args.decays), 12 * 1024
m, sq, ff, fq = 0, torch.zeros(O, dtype=torch.float64), torch.zeros(F, F, dtype=torch.float64), torch.zeros(F, O, dtype=torch.float64)
sf = torch.zeros(F, dtype=torch.float64)
for start in range(0, len(fit), 96):
    pairs = [rows(r) for r in fit[start:start + 96]]
    X, Y = torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])
    m += len(X); sf += X.sum(0); sq += Y.sum(0)
    if args.device == "cpu":
        ff += X.T @ X; fq += X.T @ Y
    else:
        for a in range(0, len(X), 32768):                                      # float32 products per slice keep rounding error far below the ridge term
            xd, yd = X[a:a + 32768].float().to(args.device), Y[a:a + 32768].float().to(args.device)
            ff += (xd.T @ xd).cpu().double(); fq += (xd.T @ yd).cpu().double()
    if start % 960 == 0:
        print("pass 2:", start + len(pairs), "records", m, "rows", round(time.time() - t0), "s", flush=True)
fm, qm = sf / m, sq / m                                                        # summaries are not exactly zero-mean: centre features too and fold into the bias
cff, cfq = ff - m * torch.outer(fm, fm), fq - m * torch.outer(fm, qm)
V = [(r, *rows(r)) for r in val]
groups = {"all": V, args.val_source: [v for v in V if str(v[0].get("source", "")).startswith(args.val_source)], "other": [v for v in V if not str(v[0].get("source", "")).startswith(args.val_source)]}
scale, best, log = float(torch.diagonal(cff).mean()), None, {}
for lam in args.ridge:
    W = torch.linalg.solve(cff + lam * scale * torch.eye(F, dtype=torch.float64), cfq)
    rel = {}
    for name, part in groups.items():
        if part:
            X, Y = torch.cat([v[1] for v in part]) - fm, torch.cat([v[2] for v in part]) - qm
            pred = X @ W
            rel[name] = [round(float((pred[:, i * 1024:(i + 1) * 1024] - Y[:, i * 1024:(i + 1) * 1024]).pow(2).mean().sqrt() / Y[:, i * 1024:(i + 1) * 1024].pow(2).mean().sqrt()), 4) for i in range(12)]
    log[str(lam)] = {k: round(float(np.mean(v)), 4) for k, v in rel.items()}
    print("ridge", lam, log[str(lam)], flush=True)
    if best is None or np.mean(rel["all"]) < best[0]:
        best = (float(np.mean(rel["all"])), lam, W, rel)
_, lam, W, rel = best
Xv, Yv = torch.cat([v[1] for v in V]) - fm, torch.cat([v[2] for v in V]) - qm
gain = Yv.std(0) / (Xv @ W).std(0).clamp_min(1e-8)
meta = {"decays": args.decays, "rank": int(P.shape[1]), "summary_scale": summary_scale, "ridge": lam, "fit_rows": m, "val_rel_rmse": rel, "sweep": log,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(), "median_gain": float(gain.median())}
np.savez(args.out, W=W.float().numpy(), b=qm.float().numpy(), gain=gain.float().numpy(), P=P, g_mean=mean32, f_mean=fm.float().numpy(), meta=np.array(json.dumps(meta)))
print(json.dumps({k: v for k, v in meta.items() if k != "val_rel_rmse"}), round(time.time() - t0), "s")
