"""Residual maps for the reverse MergeReader: what the dropped Qwen tokens carried, folded into GLM's merged entry."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, nargs="+", required=True)
parser.add_argument("--glm", type=Path, nargs="+", required=True)
parser.add_argument("--qwen", type=Path, nargs="+", required=True)
parser.add_argument("--base", type=Path, default=Path("local/live/stacked2_rev.npz"))
parser.add_argument("--filter", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rank", type=int, default=768)
parser.add_argument("--ridge", type=float, default=0.05)
parser.add_argument("--max-j", type=int, default=2)
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
gt, qt = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
base = np.load(args.base)
W0 = np.concatenate([base[f"W{l}"] for l in GL], axis=1); b0 = np.concatenate([base[f"b{l}"] for l in GL]); mean = base["g_mean"]
find = lambda folders, rid: next(f / f"{rid}.npz" for f in folders if (f / f"{rid}.npz").exists())
X = {p: {j: [] for j in range(1, args.max_j + 1)} for p in ("fit", "val")}; R = {p: {j: [] for j in range(1, args.max_j + 1)} for p in ("fit", "val")}
records, seen = [], set()
for path in args.corpus:
    for r in json.loads(path.read_text())["records"]:
        if r["split"] == "train" and r["id"] not in seen:
            seen.add(r["id"]); records.append(r)
for n, r in enumerate(records):
    part = "val" if n % 20 == 0 else "fit"
    go, qo = gt.encode(r["text"], add_special_tokens=False).offsets, qt.encode(r["text"], add_special_tokens=False).offsets
    ends = {qi for _, qi in r["aligned"]}
    merged = [(gi, qi) for gi, qi in r["aligned"] if qi - 1 >= 0 and (qi - 1) not in ends and qo[qi - 1][0] >= go[gi][0]]
    if not merged:
        continue
    g, q = np.load(find(args.glm, r["id"])), np.load(find(args.qwen, r["id"])); T = len(qo)
    G = np.concatenate([g[f"l{l}"] for l in GL], 1).astype(np.float32)
    Q = np.concatenate([np.concatenate((q[f"k{l}"].reshape(T, -1), q[f"v{l}"].reshape(T, -1)), 1) for l in QL], 1).astype(np.float32) - mean
    for gi, qi in merged:
        residual = G[gi] - (Q[qi] @ W0 + b0)
        for j in range(1, args.max_j + 1):
            if qi - j >= 0 and (qi - j) not in ends and qo[qi - j][0] >= go[gi][0]:
                X[part][j].append(Q[qi - j]); R[part][j].append(residual)
    if n % 500 == 0:
        print("scanned", n, {j: len(v) for j, v in X["fit"].items()}, flush=True)
T_ = lambda rows: torch.from_numpy(np.stack(rows)).double()
_, _, vh = torch.linalg.svd(torch.cat([T_(X["fit"][j]) for j in X["fit"] if X["fit"][j]]), full_matrices=False)
basis = vh[: args.rank].T
ridge = lambda A, Y, lam: torch.linalg.solve(A.T @ A + lam * float(torch.diagonal(A.T @ A).mean()) * torch.eye(A.shape[1], dtype=torch.float64), A.T @ Y)
out, report = {"basis": basis.float().numpy()}, {"rows": {j: len(v) for j, v in X["fit"].items()}, "val": {}}
for j in X["fit"]:
    if len(X["fit"][j]) < 200:
        continue
    # j = 2 explains what is left after j = 1 on the same merged entries; fitted independently here (the two inputs are different tokens)
    D = ridge(T_(X["fit"][j]) @ basis, T_(R["fit"][j]), args.ridge); out[f"R{j}"] = D.float().numpy()
    if X["val"][j]:
        truth = T_(R["val"][j]); report["val"][j] = float(((T_(X["val"][j]) @ basis) @ D - truth).pow(2).mean().sqrt() / truth.pow(2).mean().sqrt())
report.update({"base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(), "filter_sha256": hashlib.sha256(args.filter.read_bytes()).hexdigest(), "rank": args.rank, "ridge": args.ridge,
               "note": "val = remaining residual relative to the residual the base translator leaves on merged entries"})
np.savez(args.out, meta=np.array(json.dumps(report)), **out)
print(json.dumps(report, indent=1))
