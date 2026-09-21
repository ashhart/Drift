"""Stacked-level translators GLM -> Qwen with variance restoration (EXPLORATORY).
Every Qwen KV layer reads ALL GLM levels at the token (11 x 512 = 5632 inputs). Ridge regression
shrinks its predictions toward the mean, and the reader's attention is sensitive to exactly that
(scripts/live tolerance run), so each output dimension is rescaled to the true train-set std."""
import argparse, json
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, default=Path("local/live/corpus.json"))
parser.add_argument("--glm", type=Path, default=Path("local/live/taps_glm"))
parser.add_argument("--qwen", type=Path, default=Path("local/live/taps_qwen"))
parser.add_argument("--out", type=Path, default=Path("local/live/stacked.npz"))
parser.add_argument("--ridge", type=float, default=0.1)
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
G, Q = [], {l: [] for l in QL}
for r in json.loads(args.corpus.read_text())["records"]:
    if r["split"] != "train":
        continue
    g, q = np.load(args.glm / f"{r['id']}.npz"), np.load(args.qwen / f"{r['id']}.npz")
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    G.append(np.concatenate([g[f"l{l}"][gi].astype(np.float32) for l in GL], 1))
    for l in QL:
        Q[l].append(np.concatenate((q[f"k{l}"][qi].reshape(len(qi), -1), q[f"v{l}"][qi].reshape(len(qi), -1)), 1).astype(np.float32))
G = torch.from_numpy(np.concatenate(G)).double(); gm = G.mean(0); Gc = G - gm
gram = Gc.T @ Gc + args.ridge * len(G) * torch.eye(G.shape[1], dtype=torch.float64)
chol = torch.linalg.cholesky(gram)
out = {"g_mean": gm.float().numpy()}
for l in QL:
    q = torch.from_numpy(np.concatenate(Q[l])).double(); qm = q.mean(0)
    W = torch.cholesky_solve(Gc.T @ (q - qm), chol)
    pred = Gc @ W
    gain = (q - qm).std(0) / pred.std(0).clamp_min(1e-8)
    out[f"W{l}"], out[f"b{l}"], out[f"gain{l}"] = W.float().numpy(), qm.float().numpy(), gain.float().numpy()
    print(l, "train rel rmse", round(float((pred - (q - qm)).pow(2).mean().sqrt() / (q - qm).pow(2).mean().sqrt()), 3), "median gain", round(float(gain.median()), 2))
np.savez(args.out, **out)
