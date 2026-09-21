"""Found pool.v2 and fit one MODEL PACK per member (writer into the pool, reader out of it).

pool.v2: one vector of `--width` numbers per token. It is the top principal subspace of the founding members'
STACKED canonical entries at aligned tokens (all KV-bearing levels of every founder, concatenated). A member's
writer maps its own stacked entries into the pool; its reader maps pool rows, as written by OTHER members, to each
of its layers, with per-dimension variance restoration measured on validation rows. Everything is closed form from
streaming second moments, so a new member enrolls with one pass over the calibration text and a linear solve, and
nothing about existing members changes (H9)."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, required=True)
parser.add_argument("--glm", type=Path, required=True)
parser.add_argument("--qwen", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--width", type=int, nargs="+", default=[2048])
parser.add_argument("--ridge", type=float, default=0.03)
parser.add_argument("--val-every", type=int, default=20)
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
MEMBERS = {"glm": {"layers": GL, "width": 512, "layout": "mla_latent"}, "qwen": {"layers": QL, "width": 1024, "layout": "kv_split"}}
DG, DQ = 11 * 512, 12 * 1024
records, seen = [], set()
for r in json.loads(args.corpus.read_text())["records"]:
    if r["split"] == "train" and r["id"] not in seen:
        seen.add(r["id"]); records.append(r)
fit, val = [r for i, r in enumerate(records) if i % args.val_every], [r for i, r in enumerate(records) if not i % args.val_every]


def rows(r):
    g, q = np.load(args.glm / f"{r['id']}.npz"), np.load(args.qwen / f"{r['id']}.npz")
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    G = np.concatenate([g[f"l{l}"][gi] for l in GL], 1)
    Q = np.concatenate([np.concatenate((q[f"k{l}"][qi].reshape(len(qi), -1), q[f"v{l}"][qi].reshape(len(qi), -1)), 1) for l in QL], 1)
    return torch.from_numpy(np.concatenate((G, Q), 1).astype(np.float64))


D, n, t0 = DG + DQ, 0, time.time()
s, c = torch.zeros(D, dtype=torch.float64), torch.zeros(D, D, dtype=torch.float64)
for start in range(0, len(fit), 96):
    J = torch.cat([rows(r) for r in fit[start:start + 96]])
    n += len(J); s += J.sum(0); c += J.T @ J
    print("accumulated", start + 96, "records", n, "rows", round(time.time() - t0), "s", flush=True)
mu = s / n
c -= n * torch.outer(mu, mu)
sl = {"glm": slice(0, DG), "qwen": slice(DG, D)}
# every founder contributes equally to the subspace: scale each member's block to unit total variance before PCA
scale = torch.ones(D, dtype=torch.float64)
for m, block in sl.items():
    scale[block] = 1.0 / torch.sqrt(torch.diagonal(c)[block].sum() / n)
cs = c * scale[:, None] * scale[None, :]
evals, evecs = torch.linalg.eigh(cs)
print("eigh done", round(time.time() - t0), "s", flush=True)
Jv = torch.cat([rows(r) for r in val]) - mu
eye = lambda d: torch.eye(d, dtype=torch.float64)
for width in args.width:
    V = evecs[:, -width:].flip(1) * scale[:, None]                 # pool row z = (j - mu) @ V
    explained = float(evals[-width:].sum() / evals.sum())
    czz_target = V.T @ c @ V
    writers = {}
    for m, block in sl.items():                                    # writer: own stack -> pool target, ridge
        cmm, cmz = c[block, block], c[block] @ V
        lam = args.ridge * float(torch.diagonal(cmm).mean())
        writers[m] = torch.linalg.solve(cmm + lam * eye(cmm.shape[0]), cmz)
    out_dir = args.out / f"w{width}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(V.float().numpy().tobytes() + mu.float().numpy().tobytes()).hexdigest()
    summary = {"pool": {"version": "pool.v2", "width": width, "fingerprint": fingerprint, "explained_variance": explained, "founders": list(sl)}, "members": {}}
    for m, block in sl.items():                                    # reader: pool rows AS WRITTEN BY THE OTHER MEMBERS -> own layers
        others = [o for o in sl if o != m]
        czz, czy = torch.zeros(width, width, dtype=torch.float64), torch.zeros(width, c[block].shape[0], dtype=torch.float64)
        for o in others:
            czz += writers[o].T @ c[sl[o], sl[o]] @ writers[o]
            czy += writers[o].T @ c[sl[o], block]
        lam = args.ridge * float(torch.diagonal(czz).mean())
        reader = torch.linalg.solve(czz + lam * eye(width), czy)
        pred = torch.cat([Jv[:, sl[o]] @ writers[o] for o in others]) @ reader
        truth = torch.cat([Jv[:, block]] * len(others))
        gain = truth.std(0) / pred.std(0).clamp_min(1e-8)
        w = MEMBERS[m]["width"]
        rel = [float((pred[:, i * w:(i + 1) * w] - truth[:, i * w:(i + 1) * w]).pow(2).mean().sqrt() / truth[:, i * w:(i + 1) * w].pow(2).mean().sqrt()) for i in range(len(MEMBERS[m]["layers"]))]
        np.savez(out_dir / f"{m}.pack.npz", writer=writers[m].float().numpy(), reader=reader.float().numpy(), gain=gain.float().numpy(),
                 own_mean=mu[block].float().numpy(), meta=np.array(json.dumps({"member": m, **MEMBERS[m], "pool": summary["pool"], "ridge": args.ridge, "fit_rows": n,
                 "val_rows": int(len(Jv)), "read_rel_rmse": rel, "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest()})))
        summary["members"][m] = {"read_rel_rmse_mean": float(np.mean(rel)), "median_gain": float(gain.median())}
    (out_dir / "pool.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
