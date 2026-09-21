"""Stacked-level GLM -> Qwen translator fitted from streaming second moments (scales to millions of rows).
Each Qwen KV layer reads all 11 GLM levels at the token. Ridge shrinks predictions toward the mean and the
reader's attention cannot tolerate that (docs/agent-progress.md, tolerance run), so every output dimension is
rescaled to the true std, with the ratio measured on validation chunks that the regression never saw."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, required=True)
parser.add_argument("--glm", type=Path, required=True)
parser.add_argument("--qwen", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--ridge", type=float, nargs="+", default=[0.01, 0.1, 1.0])
parser.add_argument("--direction", choices=["glm2qwen", "qwen2glm"], default="glm2qwen")
parser.add_argument("--val-every", type=int, default=20, help="every n-th train record is validation")
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
records = [r for r in json.loads(args.corpus.read_text())["records"] if r["split"] == "train"]


seen = set()
records = [r for r in records if not (r["id"] in seen or seen.add(r["id"]))]                     # identical paragraphs occur in several doc files
fit, val = [r for i, r in enumerate(records) if i % args.val_every], [r for i, r in enumerate(records) if not i % args.val_every]
REVERSE = args.direction == "qwen2glm"


def rows(r):
    G, Q = rows_forward(r)
    return (Q, G) if REVERSE else (G, Q)


def rows_forward(r):
    g, q = np.load(args.glm / f"{r['id']}.npz"), np.load(args.qwen / f"{r['id']}.npz")
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    G = np.concatenate([g[f"l{l}"][gi] for l in GL], 1).astype(np.float64)
    Q = np.concatenate([np.concatenate((q[f"k{l}"][qi].reshape(len(qi), -1), q[f"v{l}"][qi].reshape(len(qi), -1)), 1) for l in QL], 1).astype(np.float64)
    return torch.from_numpy(G), torch.from_numpy(Q)


dg, dq, n = (12 * 1024, 11 * 512, 0) if REVERSE else (11 * 512, 12 * 1024, 0)
OUT, WIDTH = (GL, 512) if REVERSE else (QL, 1024)
sg, sq, gg, gq = torch.zeros(dg, dtype=torch.float64), torch.zeros(dq, dtype=torch.float64), torch.zeros(dg, dg, dtype=torch.float64), torch.zeros(dg, dq, dtype=torch.float64)
BATCH = 96                                                       # records per rank-k update: one large GEMM instead of many small ones
for start in range(0, len(fit), BATCH):
    pairs = [rows(r) for r in fit[start:start + BATCH]]
    G, Q = torch.cat([g for g, _ in pairs]), torch.cat([q for _, q in pairs])
    n += len(G); sg += G.sum(0); sq += Q.sum(0); gg += G.T @ G; gq += G.T @ Q
    print("accumulated", start + len(pairs), "records", n, "rows", flush=True)
gm, qm = sg / n, sq / n
cgg, cgq = gg - n * torch.outer(gm, gm), gq - n * torch.outer(gm, qm)
V = [rows(r) for r in val]
Gv, Qv = torch.cat([g for g, _ in V]) - gm, torch.cat([q for _, q in V]) - qm
scale = float(torch.diagonal(cgg).mean())
best = None
for lam in args.ridge:
    W = torch.linalg.solve(cgg + lam * scale * torch.eye(dg, dtype=torch.float64), cgq)
    pred = Gv @ W
    rel = [float((pred[:, i * WIDTH:(i + 1) * WIDTH] - Qv[:, i * WIDTH:(i + 1) * WIDTH]).pow(2).mean().sqrt() / Qv[:, i * WIDTH:(i + 1) * WIDTH].pow(2).mean().sqrt()) for i in range(len(OUT))]
    print("ridge", lam, "validation rel rmse per layer", [round(x, 3) for x in rel], "mean", round(float(np.mean(rel)), 3), flush=True)
    if best is None or np.mean(rel) < best[0]:
        best = (float(np.mean(rel)), lam, W, pred, rel)
_, lam, W, pred, rel = best
gain = Qv.std(0) / pred.std(0).clamp_min(1e-8)
out = {"g_mean": gm.float().numpy(), "meta": np.array(json.dumps({"direction": args.direction, "ridge": lam, "fit_rows": n, "val_rows": int(len(Gv)), "val_rel_rmse": rel,
       "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(), "median_gain": float(gain.median())}))}
for i, l in enumerate(OUT):
    sl = slice(i * WIDTH, (i + 1) * WIDTH)
    out[f"W{l}"], out[f"b{l}"], out[f"gain{l}"] = W[:, sl].float().numpy(), qm[sl].float().numpy(), gain[sl].float().numpy()
np.savez(args.out, **out)
print(json.loads(str(out["meta"])))
