"""Fit the fan-out maps on top of a frozen base GLM -> Qwen translator (see drift/translate/fanout.py)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, nargs="+", required=True, help="corpus json files; taps are looked up in --glm / --qwen folders in order")
parser.add_argument("--glm", type=Path, nargs="+", required=True)
parser.add_argument("--qwen", type=Path, nargs="+", required=True)
parser.add_argument("--base", type=Path, default=Path("local/live/stacked2.npz"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rank", type=int, default=768)
parser.add_argument("--ridge", type=float, default=0.05)
parser.add_argument("--max-j", type=int, default=3)
parser.add_argument("--ones", type=int, default=80000, help="how many span-1 rows to sample for the classifier")
args = parser.parse_args()
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
gt, qt = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
base = np.load(args.base)
CONTEXT = "P" in base.files                                       # a context-augmented base (drift/translate/context.py): its prediction needs the causal summaries
if CONTEXT:
    from drift.translate.context import ContextReader
    ctx = ContextReader.load(args.base, GL, QL, 2, 256); g_mean = base["g_mean"]
else:
    W0 = np.concatenate([base[f"W{l}"] for l in QL], axis=1); b0 = np.concatenate([base[f"b{l}"] for l in QL]); g_mean = base["g_mean"]
find = lambda folders, rid: next(f / f"{rid}.npz" for f in folders if (f / f"{rid}.npz").exists())
rng = np.random.default_rng(0)
X = {"fit": {j: [] for j in range(args.max_j + 1)}, "val": {j: [] for j in range(args.max_j + 1)}}      # j=0 rows feed the classifier only
R = {"fit": {j: [] for j in range(1, args.max_j + 1)}, "val": {j: [] for j in range(1, args.max_j + 1)}}
N = {"fit": [], "val": []}
records, seen = [], set()
for path in args.corpus:
    for r in json.loads(path.read_text())["records"]:
        if r["split"] == "train" and r["id"] not in seen:
            seen.add(r["id"]); records.append(r)
for n, r in enumerate(records):
    part = "val" if n % 20 == 0 else "fit"
    go, qo = gt.encode(r["text"], add_special_tokens=False).offsets, qt.encode(r["text"], add_special_tokens=False).offsets
    g, q = np.load(find(args.glm, r["id"])), np.load(find(args.qwen, r["id"]))
    G = np.concatenate([g[f"l{l}"] for l in GL], 1).astype(np.float32) - g_mean
    predicted = ctx.features({l: g[f"l{l}"] for l in GL})[0] @ ctx.weights + ctx.bias if CONTEXT else None
    Q = np.concatenate([np.concatenate((q[f"k{l}"].reshape(len(qo), -1), q[f"v{l}"].reshape(len(qo), -1)), 1) for l in QL], 1).astype(np.float32)
    end_of = {gi: qi for gi, qi in r["aligned"]}
    spans = {gi: 1 for gi in end_of}
    for gi, qi in r["aligned"]:                                   # reader tokens that lie inside this writer token, before its endpoint
        j = 1
        while qi - j >= 0 and qo[qi - j][0] >= go[gi][0] and qo[qi - j][1] <= go[gi][1] and (qi - j) not in end_of.values() and j <= args.max_j:
            X[part][j].append(G[gi]); R[part][j].append(Q[qi - j] - (predicted[gi] if CONTEXT else G[gi] @ W0 + b0)); spans[gi] = j + 1; j += 1
    for gi, n_span in spans.items():
        if n_span > 1 or rng.random() < 0.2:
            X[part][0].append(G[gi]); N[part].append(n_span)
    if n % 400 == 0:
        print("scanned", n, {j: len(v) for j, v in X["fit"].items()}, flush=True)
T = lambda rows: torch.from_numpy(np.stack(rows)).double()
Xc, Nc = T(X["fit"][0]), np.array(N["fit"])
keep = np.concatenate((np.where(Nc > 1)[0], rng.choice(np.where(Nc == 1)[0], size=min(args.ones, int((Nc == 1).sum())), replace=False)))
everything = torch.cat([Xc[keep]] + [T(X["fit"][j]) for j in range(1, args.max_j + 1) if X["fit"][j]])
_, _, vh = torch.linalg.svd(everything - 0 * everything.mean(0), full_matrices=False)
basis = vh[: args.rank].T                                          # [inputs, r]
ridge = lambda A, Y, lam: torch.linalg.solve(A.T @ A + lam * float(torch.diagonal(A.T @ A).mean()) * torch.eye(A.shape[1], dtype=torch.float64), A.T @ Y)
classes = args.max_j + 1
Zc, onehot = Xc[keep] @ basis, torch.nn.functional.one_hot(torch.from_numpy(np.minimum(Nc[keep], classes) - 1), classes).double()
weights = torch.where(torch.from_numpy(Nc[keep] > 1), 4.0, 1.0).double().sqrt()[:, None]          # span>1 rows are rare: weight them up
mean_z, mean_y = Zc.mean(0), onehot.mean(0)
cw = ridge((Zc - mean_z) * weights, (onehot - mean_y) * weights, args.ridge); cb = mean_y - mean_z @ cw
Zv, Nv = T(X["val"][0]) @ basis, np.minimum(np.array(N["val"]), classes)
scores = Zv @ cw + cb
report = {"rows": {j: len(v) for j, v in X["fit"].items()}, "val_rows": {j: len(v) for j, v in X["val"].items()}, "classifier": {}}
for margin in (0.0, 0.1, 0.2, 0.3):
    s = scores.clone(); s[:, 1:] -= margin; pred = s.argmax(1).numpy() + 1
    multi, pmulti = Nv > 1, pred > 1
    report["classifier"][margin] = {"exact": float((pred == Nv).mean()), "precision_multi": float((multi & pmulti).sum() / max(1, pmulti.sum())), "recall_multi": float((multi & pmulti).sum() / max(1, multi.sum())),
                                    "exact_on_multi": float((pred[multi] == Nv[multi]).mean())}
best = max(report["classifier"], key=lambda m: report["classifier"][m]["exact"])
out = {"basis": basis.float().numpy(), "count_w": cw.float().numpy(), "count_b": cb.float().numpy()}
report["residual"] = {}
for j in range(1, args.max_j + 1):
    if len(X["fit"][j]) < 200:
        continue
    Zj, Rj = T(X["fit"][j]) @ basis, T(R["fit"][j])
    D = ridge(Zj, Rj, args.ridge)
    out[f"R{j}"] = D.float().numpy()
    if X["val"][j]:
        Zvj, Rvj = T(X["val"][j]) @ basis, T(R["val"][j])
        truth = Rvj                                              # residual target: what the base map misses for this reader token
        report["residual"][j] = {"rows": len(X["fit"][j]), "val_rel_error_base_only": 1.0, "val_rel_error_with_residual": float((Zvj @ D - truth).pow(2).mean().sqrt() / truth.pow(2).mean().sqrt())}
report.update({"margin": best, "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(), "rank": args.rank, "ridge": args.ridge})
np.savez(args.out, meta=np.array(json.dumps(report)), **out)
print(json.dumps(report, indent=1))
